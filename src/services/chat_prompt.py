"""ChatML-style and strict two-role message preparation for local tokenizers."""


class ChatTemplateError(ValueError):
    """The selected template cannot retain the required conversation structure."""


class ChatPromptAdapter:
    """Adapt canonical text messages; the tokenizer owns model-specific formatting."""

    USER_SEPARATOR = '\n\n--- Next user message ---\n\n'

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.profile = None  # Resolved to 'chatml' or 'two_role_strict' after validate() runs

        try:
            self.template = tokenizer.get_chat_template()
        except ValueError as error:
            raise ValueError(
                'The model needs matching tokenizer files with a default chat template.'
            ) from error

        if not isinstance(self.template, str) or not self.template.strip():
            raise ValueError('The tokenizer chat template is empty; use the matching chat-model tokenizer')

    def adapt(self, messages):
        """Resolve compatibility once and adapt a copy without changing history."""
        self.validate()
        return self._adapt(messages, self.profile)

    def _adapt(self, messages, profile):
        """Apply one profile to a copy, including during startup probes."""
        copied = []
        for index, message in enumerate(messages):
            role, content = message.get('role'), message.get('content')
            if role not in ('system', 'user', 'assistant') or not isinstance(content, str):
                raise ValueError('Local chat requires system/user/assistant roles with text content')
            # Enforce that system instructions may only exist at the very beginning of the prompt
            if role == 'system' and index != 0:
                raise ValueError('Local chat allows only one initial system message')
            copied.append({'role': role, 'content': content})

        # Separate initial system message from turn-based dialogue
        system = copied.pop(0) if copied and copied[0]['role'] == 'system' else None
        # Drop empty/whitespace-only system prompts so they don't consume prompt budget
        if system is not None and not system['content'].strip():
            system = None

        # Guard against malformed dialogue sequences
        if not copied or copied[0]['role'] != 'user' or copied[-1]['role'] != 'user':
            raise ValueError('Local chat must start with a user turn and end with the current user input')

        turns = []
        for message in copied:
            # Under strict 2-role schemas (e.g. Gemma), consecutive user turns caused by barge-in
            # or partial utterances must be coalesced into a single turn block.
            if (profile == 'two_role_strict' and message['role'] == 'user'
                    and turns and turns[-1]['role'] == 'user'):
                turns[-1]['content'] += self.USER_SEPARATOR + message['content']
            else:
                turns.append(message)

        if system is not None:
            if profile == 'two_role_strict':
                # 2-role workaround: embed system prompt into the first user turn
                turns[0]['content'] = system['content'] + '\n\n' + turns[0]['content']
            else:
                # ChatML: preserve system instruction as a native independent turn
                turns.insert(0, system)

        return turns

    def _render(self, messages, profile):
        """Apply Jinja formatting to the adapted message payload."""
        adapted = self._adapt(messages, profile)
        options = {
            'tokenize': False,
            'add_generation_prompt': True,
            'chat_template': self.template
        }
        # Differentiate between broken Jinja files and model schema rejections
        from jinja2 import TemplateError, TemplateSyntaxError
        try:
            return self.tokenizer.apply_chat_template(adapted, **options)
        except TemplateSyntaxError as error:
            raise ValueError('Local chat template has invalid Jinja syntax; use the matching chat-model tokenizer') from error
        except TemplateError as error:
            raise ChatTemplateError('Local chat template rejected the conversation') from error

    def tokenize(self, messages):
        """Render and tokenize one candidate after canonical history pruning."""
        self.validate()
        rendered = self._render(messages, self.profile)
        return self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False)

    def validate(self):
        """Select one of two profiles at startup and keep it for this adapter."""
        if self.profile is not None:
            return
        last_error = None
        for profile in ('chatml', 'two_role_strict'):
            try:
                self._validate_candidate(profile)
            except ChatTemplateError as error:
                last_error = error
                continue
            self.profile = profile
            return
        raise ChatTemplateError(
            'The model template cannot represent this conversation using ChatML-style or strict '
            'two-role messages. Use a supported chat model with its matching tokenizer files.'
        ) from last_error

    def _validate_candidate(self, profile):
        """Check a first question and an interrupted follow-up without inference."""
        user = {'role': 'user', 'content': 'prompt_probe_user_one'}
        assistant = {'role': 'assistant', 'content': 'prompt_probe_assistant_one'}
        current = {'role': 'user', 'content': 'prompt_probe_user_two'}
        followup = {'role': 'user', 'content': 'prompt_probe_user_three'}
        system = {'role': 'system', 'content': 'prompt_probe_system'}

        for turns in ([user], [user, assistant, current, followup]):
            probe = [system, *turns]
            rendered = self._render(probe, profile)
            markers = [message['content'] for message in probe]
            # Ensure template did not silently drop message bodies
            if not isinstance(rendered, str) or any(marker not in rendered for marker in markers):
                raise ChatTemplateError('Local chat template discarded probe content')

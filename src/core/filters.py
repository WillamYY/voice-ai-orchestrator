"""Stateless speech cleanup for completed segments."""


class SpeechTextFilter:
    """
    Literal matching for optional symbol/emoji removal within one segment.

    This is speech cleanup, not a Markdown parser or a content-safety filter.
    Matches are case-sensitive substrings. Prefer the longest matching literal.
    No text is retained between calls, so matches cannot span segments.
    """

    MARKDOWN_SYMBOLS = ('*', '_', '`', '~', '#', '>')
    EMOTICONS = (':-)', ':)', ':-(', ':(', ';-)', ';)', ':-D', ':D',
                 ':-P', ':P', '<3', '^_^', '(^_^)')

    def __init__(self, *, markdown=True, emoji=True, emoticons=True, remove=()):
        patterns = set(remove)
        if any(not isinstance(item, str) or not item for item in patterns):
            raise ValueError('Filter removal entries must be nonempty strings')
        if markdown:
            patterns.update(self.MARKDOWN_SYMBOLS)
        if emoticons:
            patterns.update(self.EMOTICONS)
        if emoji:
            patterns.update(char + suffix for char in '0123456789#*'
                            for suffix in ('\u20e3', '\ufe0f\u20e3'))
        self._patterns = {}
        for pattern in sorted(patterns, key=len, reverse=True):
            self._patterns.setdefault(pattern[0], []).append(pattern)
        self._emoji = emoji

    @staticmethod
    def _is_emoji_character(char: str) -> bool:
        # Common pictographs, flags, modifiers, dingbats and sequence controls.
        # These ranges are deliberately documented as a heuristic, not a full
        # Unicode emoji database (some symbols also have non-emoji uses).
        value = ord(char)
        return (0x1F000 <= value <= 0x1FAFF or 0x2600 <= value <= 0x27BF
                or 0xE0020 <= value <= 0xE007F
                or char in '\u200d\ufe0e\ufe0f\u20e3')

    def clean(self, text: str) -> str:
        """Clean one completed segment without changing segmentation boundaries."""
        output = []
        position = 0
        while position < len(text):
            char = text[position]
            candidates = self._patterns.get(char, ())
            match = next((pattern for pattern in candidates if text.startswith(pattern, position)), None)
            if match is not None:
                position += len(match)
            else:
                if not self._emoji or not self._is_emoji_character(char):
                    output.append(char)
                position += 1
        return ''.join(output)

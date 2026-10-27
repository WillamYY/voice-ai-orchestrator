"""Deterministic incremental speech segmentation."""
import re

from src.core.filters import SpeechTextFilter


class TextSegmenter:
    def __init__(self, max_chars=None, *, config=None):
        # Internal accumulator storing tokens that haven't formed a complete clause yet
        self.buffer = ''
        
        # Upper length boundary for a spoken segment.
        # Keeps individual TTS synthesis requests bounded to preserve low latency
        self.max_chars = max_chars if max_chars is not None else (config.segment_chars if config is not None else 120)
        if self.max_chars < 1:
            raise ValueError('max_chars must be positive')
        self._filter = None
        if config is not None and config.text_filter_enabled:
            self._filter = SpeechTextFilter(
                markdown=config.text_filter_markdown, emoji=config.text_filter_emoji,
                emoticons=config.text_filter_emoticons, remove=config.text_filter_remove,
            )

    def feed(self, text):
        """Append newly arrived text/tokens and extract all completed sentence segments."""
        self.buffer += text
        return self._extract(final=False)

    def _sentence_end(self, *, final):
        for match in re.finditer(r'[.!?;。！？；\n]', self.buffer):
            index = match.start()
            if match.group() == '.' and index > 0 and self.buffer[index - 1].isdecimal():
                if index + 1 == len(self.buffer) and not final:
                    # Wait for one more character (or finish) to distinguish 3.14
                    # from a sentence ending in 3. across LLM chunk boundaries.
                    return None
                if index + 1 < len(self.buffer) and self.buffer[index + 1].isdecimal():
                    continue
            return match.end()
        return None

    def _extract(self, *, final):
        result = []
        
        while self.buffer:
            # Punctuation Boundary Check:
            # Look for sentence-ending delimiters:
            sentence_end = self._sentence_end(final=final)
            
            # If punctuation is found within the allowed character budget, split right after it
            if sentence_end is not None and sentence_end <= self.max_chars:
                end = sentence_end
                
            # If it's a very long sentence, forcibly split the sentence to prevent audio generation from stalling
            elif len(self.buffer) >= self.max_chars:
                # Split at the last space before max_chars to avoid cutting words in half
                # rfind() returns the index of the target; if no target, return -1
                end = self.buffer.rfind(' ', 0, self.max_chars)
                # If a space exists, slice after it; otherwise hard cut at max_chars :(
                end = end + 1 if end > 0 else self.max_chars

            elif final:
                end = len(self.buffer)
                
            # If not enough text or no delimiter yet, no segmenting process and wait for more tokens
            else:
                break
            
            # Extract the complete segment and retain remainder in self.buffer for next iteration
            segment, self.buffer = self.buffer[:end], self.buffer[end:]
            if self._filter is not None:
                segment = self._filter.clean(segment)
            
            # Discard segments with only whitespace
            if segment.strip():
                result.append(segment)
                
        return result

    def finish(self):
        """Flush any trailing text remaining in the buffer when the LLM stream terminates."""
        # Yield trailing text only if it contains non-whitespace characters
        return self._extract(final=True)

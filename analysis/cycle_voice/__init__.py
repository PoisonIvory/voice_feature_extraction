"""Independent N-of-1 voice-cycle analysis layer.

This package is a deliberately separate analysis layer that consumes the
extraction pipeline's voice-feature artifacts plus staged external biometric
inputs. It does not import from or modify the extraction package, keeping the
data-preparation project (see USER_STORIES.md) clean.
"""

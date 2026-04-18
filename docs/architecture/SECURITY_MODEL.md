# Security Model

## Threat Model
This section discusses the potential security threats and mitigation strategies for Xylia.

## Areas of Concern
- **API Key Exposure**: Ensure `GEMINI_API_KEY` is securely stored in `.streamlit/secrets.toml` or environment variables, not committed to source control.
- **Input Validation**: All user-uploaded content must be sanitized and validated before processing to prevent malicious file execution.
- **Data Privacy**: Ensure that session data stored in TinyDB does not leak across different user sessions. Implement strict isolation for local processing.

## Best Practices
- Regular dependency scanning.
- Implementation of rate limiting to prevent abuse of the Gemini Vision API.

# REST API Specification

## Overview
This document specifies the endpoints exposed by the Xylia backend services (for future decoupling from Streamlit).

## Endpoints

### `POST /api/v1/analyze`
**Description**: Analyzes an uploaded image.
**Payload**: `multipart/form-data` containing the image.
**Response**:
```json
{
  "status": "success",
  "data": {
    "category": "plant",
    "confidence": 0.95,
    "analysis": "..."
  }
}
```

### `GET /api/v1/history`
**Description**: Retrieves the analysis history for the current session.
**Response**:
```json
{
  "status": "success",
  "data": [
    {
      "id": "uuid",
      "timestamp": "2026-03-15T...",
      "summary": "..."
    }
  ]
}
```

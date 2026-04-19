# WebSocket Events Specification

## Overview
Defines the real-time events for long-running analysis tasks or collaborative sessions.

## Events

### `Client -> Server`
- `START_ANALYSIS`: Trigger analysis process.
  - Payload: `{ image_id: "uuid" }`
- `SUBSCRIBE_SESSION`: Join a collaborative study session.
  - Payload: `{ session_id: "uuid" }`

### `Server -> Client`
- `ANALYSIS_PROGRESS`: Updates on the processing state.
  - Payload: `{ status: "extracting_features", progress: 50 }`
- `ANALYSIS_COMPLETE`: Final results notification.
  - Payload: `{ result_id: "uuid" }`
- `ERROR`: Error notification.
  - Payload: `{ code: 500, message: "API timeout" }`

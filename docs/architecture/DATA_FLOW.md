# Data Flow Specification

## Overview
This document describes how data propagates through the Xylia system.

## Ingestion Pipeline
1. Client uploads an image via the Streamlit UI.
2. The UI layer validates the image format (PNG, JPEG).
3. The image is passed to the PIL/Pillow component for basic processing (resizing, format normalization).

## Processing Pipeline
1. Processed image is converted to a base64 string or an optimized format.
2. The payload is sent to the Gemini Vision API alongside contextual text prompts.
3. The response is parsed and structured.

## Storage Pipeline
1. Resulting data (classification, descriptions) is sent to TinyDB.
2. The user's session history is updated to reflect the new interaction.

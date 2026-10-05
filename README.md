# Signa — Real-Time Sign Language Recognition

A browser-based real-time sign language recognition application powered by a trained deep-learning image classifier.

## Features

- Live webcam recognition
- Real-time sign prediction
- Confidence display
- Browser-based camera access
- Responsive, product-style interface
- Designed for deployment as a public web application

## Project structure

```text
sign-language-detector/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── sign_language_final_41class.keras
│   └── sign_language_classes_41.json
└── assets/
```

## Run locally

Create/activate your Python environment, install dependencies, and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL in your browser and allow camera access.

## Deployment

This application is designed for a hosted Streamlit environment. The browser requests camera permission and sends the live video stream to the application for recognition.

Before deployment, make sure both files below are present inside `models/`:

- `sign_language_final_41class.keras`
- `sign_language_classes_41.json`

## Notes

The public-facing interface intentionally keeps implementation details out of the product UI. Technical details, training information, and architecture can be documented here in the repository instead.

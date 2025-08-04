# UI Test Environment

This folder contains Streamlit prototypes and UI experiments for the improved-local-assistant webapp.

## Structure
- `streamlit_prototypes/` - Individual Streamlit app prototypes
- `mock_data/` - Sample data for testing UI components
- `components/` - Reusable UI components
- `assets/` - Images, CSS, and other static assets for prototypes

## Usage

### Running a prototype:
```bash
cd ui-test/streamlit_prototypes
streamlit run chat_interface.py
```

### Creating new prototypes:
1. Create a new .py file in `streamlit_prototypes/`
2. Use mock data from `mock_data/` for testing
3. Test components before integrating into main app

## Notes
- This is a sandbox environment for UI experimentation
- Finalized designs should be integrated into the main app structure
- Keep prototypes lightweight and focused on specific UI elements
Note: The weights file is approximately 250MB.

## Usage

Run the main script:
\`\`\`
python vehicle_detection.py
\`\`\`

- Press 'q' to quit the application
- Follow on-screen prompts to save the output video

## Configuration

You can adjust detection parameters in `config/settings.py`:
- Confidence threshold
- Non-maximum suppression threshold
- Vehicle classes to detect
- Speed threshold for fast/slow classification
- Counting line position

## License

MIT License
\`\`\`

### 2. requirements.txt

```txt project="Vehicle Detection System" file="requirements.txt" type="code"
opencv-python==4.8.0
numpy==1.24.3
# Model Saving and Loading Guide

This guide explains how to save and load models in the brAin project.

## During Training

When you run the training command, models are automatically saved at regular intervals and at the end of training.

### Training Command

To train a model and save it:

```bash
python main.py train --scenario basic --episodes 1000 --model results/my_model --output-dir results/basic_training
```

This command will:
1. Train the agent for 1000 episodes on the "basic" scenario
2. Save intermediate models every 100 episodes (configurable in trainer_config)
3. Save the final model to `results/basic_training/my_model`

The saved model consists of two files:
- `my_model_state.pkl`: Contains agent parameters and metrics
- `my_model_components.pkl`: Contains trained components (SNN, memory, etc.)

### Important Parameters

- `--model`: Base filename for saved model
- `--output-dir`: Directory to save results (models, metrics, plots)
- `--episodes`: Number of episodes to train
- `--use-yolo`: Enable YOLO object detection

### Auto-Save Frequency

The model is saved:
- Every 100 episodes (configurable via `save_frequency` in `trainer_config`)
- After training completes

## Loading Models for Testing

To test a trained model:

```bash
python main.py test --scenario basic --model results/basic_training/my_model --render
```

The system will:
1. Look for model files at the specified path
2. If not found, it will try to find them in the `output-dir` directory
3. Load the model if found, or display an error message if not found

### Troubleshooting Loading Issues

If you encounter issues loading a model:

1. **Check file paths**: Ensure the model files exist at the specified path
2. **Check file extensions**: The system expects `_state.pkl` and `_components.pkl` suffix
3. **Python version**: Ensure you're using the same Python version that created the model
4. **Dependencies**: Ensure all dependencies are installed, especially for YOLO models

## Model Components

The brAin agent model consists of several components:

- **HDCEncoder**: Hyperdimensional Computing encoder for visual processing
- **SpikingNeuralNetwork**: Neural network for action selection
- **CellularAutomata**: Pattern processing component
- **EpisodicMemory**: Experience storage system
- **SemanticMemory**: Knowledge consolidation system
- **YOLO Detector**: Object detection component (if enabled)

## YOLO Model Considerations

When using YOLO object detection:

1. The YOLO model itself is not saved in the pickle files to avoid compatibility issues
2. Instead, the HDCEncoder is recreated with the saved parameters during loading
3. A new YOLO detector is initialized when the model is loaded

This ensures compatibility and stability when working with PyTorch-based YOLO models.

## Example Workflow

A typical workflow might look like:

1. **Train on basic scenario**:
   ```bash
   python main.py train --scenario basic --episodes 1000 --model basic_agent --output-dir results/basic
   ```

2. **Test the trained model**:
   ```bash
   python main.py test --scenario basic --model results/basic/basic_agent --render
   ```

3. **Visualize internal representations**:
   ```bash
   python main.py visualize --scenario basic --model results/basic/basic_agent --visualize-internals --render
   ```

4. **Evaluate generalization**:
   ```bash
   python main.py eval --model results/basic/basic_agent --output-dir results/evaluation
   ```

## Backing Up Models

It's recommended to backup your trained models, especially after significant training time:

```bash
cp -r results/basic/basic_agent* backups/
```

## Known Limitations

- Large models may take significant disk space and time to save/load
- YOLO detector state needs to be reinitialized on load, so exact detection behavior may vary slightly 
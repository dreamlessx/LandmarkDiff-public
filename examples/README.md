# Examples

Example scripts and notebooks demonstrating LandmarkDiff functionality.

## Scripts

| Example | Description | GPU Required |
|---------|-------------|:---:|
| [basic_inference.py](basic_inference.py) | Single image prediction | Yes |
| [batch_inference.py](batch_inference.py) | Process multiple images | Yes |
| [tps_only.py](tps_only.py) | CPU-only geometric deformation | No |
| [compare_procedures.py](compare_procedures.py) | Side-by-side procedure comparison | Yes |
| [custom_procedure.py](custom_procedure.py) | Define and use a custom procedure | No |
| [landmark_visualization.py](landmark_visualization.py) | Visualize face mesh and deformations | No |
| [evaluate_checkpoint.py](evaluate_checkpoint.py) | Evaluate a trained checkpoint | Yes |

## Running examples

```bash
# Install dependencies
pip install -e ".[app]"

# Run an example
python examples/basic_inference.py --image path/to/face.jpg
```

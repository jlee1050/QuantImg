# QuantImg
Learn quantum gates by seeing them, one pixel at a time.

Upload an image, build a 2-qubit circuit (X, Y, Z, H, CNOT), and see how each gate transforms the pixels.
Each pair of neighbouring pixels becomes two qubits: brightness is encoded with an RY rotation,
the circuit is applied with Qiskit, and measurement results (with adjustable shots) become the new brightness.

Built with Qiskit and Gradio for IBM Qiskit Fall Fest 2026 (UW Quantum Club).
By Juyeon Lee and Aarnav Thite.

## Run
Open `Quantum_Image_Manipulation.ipynb` in Google Colab and run all cells.
The last cell prints a public Gradio link to the app.


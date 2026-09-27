# Neural-network lab

A browser UI and Python examples for seeing how a small neural network learns. Try different activation functions, learning rates and datasets.

## Browser UI

Use Node.js 22.12 or newer. From this folder:

```powershell
cd nn-app
npm ci
npm run dev
```

Open the URL printed by Vite. `npm run build` creates a static build; `npm run preview` serves it locally.

## Python examples

```powershell
python nn_visualizer.py
python nn_known_data.py
```

`nn_visualizer.py` opens a Tkinter window. `nn_known_data.py` prints a worked training example. Both use the Python standard library. If your Python installation omits Tkinter, install the Tk package supplied by your OS.

## Docker

From this folder:

```powershell
docker build -t nn-lab .
docker run --rm -p 8080:80 nn-lab
```

Open http://localhost:8080. The container builds the same `nn-app` source used by the development command. `neural-network-lab.jsx` is an earlier standalone version retained for comparison.

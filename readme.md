Here's a `README.md` that covers the steps for editing the Docker command, attaching VS Code to the container, and selecting the correct Python environment:

# Rapid Startup Environment Setup for Hydra-Link EMNLP 2025

This README provides a step-by-step guide to setting up a Docker container for rapid development of the Hydra-Link project, ensuring a smooth environment for all development tasks and easy attachment of Visual Studio Code (VS Code) to the container.

## Prerequisites

Before starting, ensure that the following are installed:

1. Docker (with GPU support enabled).
2. NVIDIA drivers for your GPU (with CUDA compatibility).
3. Visual Studio Code (VS Code) with the Remote - Containers extension installed.

## 1. Build and Run the Docker Container

This step involves running a Docker container with the correct volume mounts and exposed ports.

### 1.1 Edit and Run the Docker Command

Run the following command in your terminal to create and start the Docker container. The container will mount your local project directory into the container's `/workspace` directory, allowing you to access your code inside the container.

```bash
docker run --gpus all -v "C:/Documents/All_github_repo/PKB1/4. Projects (Non-repeatable that build from 3.x)/@Hydra-link (EMNLP2025 submission)/lab:/workspace" -p 8888:8888 -p 6006:6006 --name rapid-startup-env -it nvcr.io/nvidia/pytorch:25.04-py3
```

Explanation of the command:

- `--gpus all`: Allocates all available GPUs to the container.
- `-v "C:/path/to/your/project:/workspace"`: Mounts your local project directory (`lab`) to `/workspace` inside the container.
- `-p 8888:8888 -p 6006:6006`: Exposes ports for Jupyter Notebook (8888) and TensorBoard (6006).
- `--name rapid-startup-env`: Names the container `rapid-startup-env` for easy reference.
- `-it`: Runs the container interactively with a terminal.
- `nvcr.io/nvidia/pytorch:25.04-py3`: The NVIDIA PyTorch image with CUDA support.

### 1.2 Verify the Docker Container

After running the command, verify that the container is running by listing all running containers:

```bash
docker ps
```

You should see your `rapid-startup-env` container listed in the output.

## 2. Attach VS Code to the Docker Container

Visual Studio Code has a Remote - Containers extension that allows you to open and work directly within a Docker container.

### 2.1 Install the Remote - Containers Extension

If you haven't already, install the Remote - Containers extension for Visual Studio Code. You can find it in the VS Code marketplace.

- Open VS Code
- Go to the Extensions view (`Ctrl+Shift+X`)
- Search for Remote - Containers
- Click Install

### 2.2 Attach VS Code to the Container

Once the extension is installed:

1. Open VS Code.
2. Open the Command Palette (`Ctrl+Shift+P`).
3. Type and select Remote-Containers: Attach to Running Container....
4. Select the `rapid-startup-env` container.

VS Code will now open the project from your Docker container. You can edit files as if they are on your local machine.

## 3. Select the Correct Python Interpreter

After attaching to the container, you must ensure that VS Code uses the correct Python interpreter that corresponds to the Python version inside the container.

### 3.1 Select Python Interpreter in VS Code

1. Open the Command Palette (`Ctrl+Shift+P`).
2. Type and select Python: Select Interpreter.
3. You should see a list of available Python interpreters. Select the one that corresponds to your mounted environment. It should look something like this:

   ```
   /workspace/venv/bin/python
   ```

   If the virtual environment is already set up inside the container, it will be listed here.

### 3.2 Check Python Version

To verify that the correct Python version is being used:

1. Open the terminal in VS Code (`Ctrl+`).
2. Run:
   ```bash
   python --version
   ```

   Ensure it shows the correct version, for example, Python 3.10.x.

## 4. Stopping the Container

To stop the container when you're done working:

1. Exit the container by typing `exit` in the container terminal.
2. Stop the container:
   ```bash
   docker stop rapid-startup-env
   ```
3. Optionally, remove the container to free up resources:
   ```bash
   docker rm rapid-startup-env
   ```

## 5. Troubleshooting

- GPU not detected: Ensure that your system has the correct NVIDIA drivers and that the `--gpus all` flag is included in the Docker run command.
- VS Code does not recognize Python environment: Make sure you have selected the correct Python interpreter by following the steps under section 3.

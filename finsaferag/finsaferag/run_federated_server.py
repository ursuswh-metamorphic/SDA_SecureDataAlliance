# run_federated_server.py

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context
from flwr.server.run_serverapp import run

# Import server_app from your project
from server_app import app as flower_app


def server_fn(context: Context) -> ServerAppComponents:
    """
    Flower requires server_fn returning ServerAppComponents.
    num_rounds must be big to prevent shutdown.
    """
    server_config = ServerConfig(num_rounds=999999)

    return ServerAppComponents(
        server=flower_app,
        server_config=server_config,
    )


if __name__ == "__main__":
    print("🚀 Starting Flower Server (real mode, no simulation)...")

    # Create ServerApp
    server_app = ServerApp(server_fn=server_fn)

    # Run using correct API for flwr 1.23.0
    run(server_app)

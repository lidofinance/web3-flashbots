from typing import Callable, Any
from web3 import Web3
from web3.middleware import Middleware
from web3.types import RPCEndpoint, RPCResponse
from .provider import FlashbotProvider


FLASHBOTS_METHODS = [
    "eth_sendBundle",
    "eth_callBundle",
    "flashbots_getUserStats",
    "flashbots_getBundleStats",
]


def construct_flashbots_middleware(
    flashbots_provider: FlashbotProvider,
) -> Middleware:
    """
    Captures Flashbots RPC requests and sends them to the Flashbots endpoint
    while also injecting the required authorization headers
    Keyword arguments:
    flashbots_provider -- An HTTP provider instantiated with any authorization headers
    required
    """

    def flashbots_middleware(
        make_request: Callable[[RPCEndpoint, Any], Any],
        w3: Web3,  # pylint: disable=W0613
    ) -> Callable[[RPCEndpoint, Any], RPCResponse]:
        def middleware(method: RPCEndpoint, params: Any) -> RPCResponse:
            if method not in FLASHBOTS_METHODS:
                return make_request(method, params)

            # otherwise intercept it and POST it
            return flashbots_provider.make_request(method, params)

        return middleware

    return flashbots_middleware

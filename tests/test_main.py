import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import asyncio

# Import main module to access its globals
import main
from main import app, validate_configuration


class TestConfiguration:
    def test_validate_configuration_with_defaults(self):
        """Test configuration validation with default values."""
        with patch('main.logger') as mock_logger:
            validate_configuration()
            mock_logger.warning.assert_any_call("LETTA_BASE_URL is using default value. Please set LETTA_BASE_URL environment variable.")
            mock_logger.warning.assert_any_call("LETTA_API_KEY not set. Authentication may fail for Letta Cloud.")

    def test_validate_configuration_with_custom_url(self):
        """Test configuration validation with custom URL."""
        # Test with custom URL (simulating environment variable)
        original_url = main.LETTA_BASE_URL
        try:
            # Temporarily change the global variable
            main.LETTA_BASE_URL = 'https://custom.letta.com'
            with patch('main.logger') as mock_logger:
                validate_configuration()
                mock_logger.warning.assert_called_once_with("LETTA_API_KEY not set. Authentication may fail for Letta Cloud.")
                # Verify the default URL warning was NOT called
                assert mock_logger.warning.call_count == 1, "Should only call warning once (for API key)"
                assert "LETTA_BASE_URL is using default value" not in str(mock_logger.warning.call_args_list)
        finally:
            # Restore original value
            main.LETTA_BASE_URL = original_url


class TestHealthCheck:
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "letta_base_url" in data
        assert "letta_connected" in data
        assert "agents_loaded" in data


class TestMessagePreparation:
    @pytest.mark.asyncio
    async def test_prepare_user_message(self):
        """Test user message preparation."""
        from main import _prepare_message
        from letta_client.types import TextContent

        message = {"role": "user", "content": "Hello, world!"}
        result = await _prepare_message(message)

        assert result.role == "user"
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_prepare_unsupported_role(self):
        """Test unsupported message role."""
        from main import _prepare_message
        from fastapi import HTTPException

        message = {"role": "system", "content": "System message"}

        with pytest.raises(HTTPException) as exc_info:
            await _prepare_message(message)

        assert exc_info.value.status_code == 400
        assert "Unsupported message role: system" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_prepare_unsupported_role(self):
        """Test unsupported message role."""
        from main import _prepare_message
        from fastapi import HTTPException

        message = {"role": "system", "content": "System message"}

        with pytest.raises(HTTPException) as exc_info:
            await _prepare_message(message)

        assert exc_info.value.status_code == 400
        assert "Unsupported message role" in str(exc_info.value.detail)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
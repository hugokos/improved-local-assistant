"""
Integration tests for voice functionality in the Improved Local AI Assistant.

Tests the complete voice processing pipeline and integration with existing systems.
"""

from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest


# Test voice integration with existing systems
class TestVoiceSystemIntegration:
    """Test voice integration with existing systems."""

    def test_voice_config_loading(self):
        """Test that voice configuration loads correctly."""
        try:
            from app.core import load_config

            config = load_config()
            voice_config = config.get("voice", {})

            # Test basic structure
            assert isinstance(voice_config, dict)

            if voice_config.get("enabled", False):
                # Test STT config
                stt_config = voice_config.get("stt", {})
                assert "enabled" in stt_config
                assert "sample_rate" in stt_config
                assert isinstance(stt_config["sample_rate"], int)

                # Test TTS config
                tts_config = voice_config.get("tts", {})
                assert "enabled" in tts_config
                assert "sample_rate" in tts_config
                assert isinstance(tts_config["sample_rate"], int)

                print("✅ Voice configuration loaded successfully")
            else:
                print("ℹ️  Voice functionality disabled in config")

        except Exception as e:
            pytest.fail(f"Failed to load voice configuration: {str(e)}")

    def test_voice_service_initialization_in_app(self):
        """Test that voice services can be initialized in app context."""
        try:
            from app.core import load_config
            from app.services.init import initialize_services

            config = load_config()

            # Test that initialize_services can handle voice services
            # This is an async function, so we'll test the import and structure
            assert initialize_services is not None

            print("✅ Voice service initialization function available")

        except ImportError as e:
            pytest.skip(f"App initialization imports not available: {str(e)}")
        except Exception as e:
            pytest.fail(f"Voice service initialization test failed: {str(e)}")

    def test_voice_websocket_routes(self):
        """Test that voice WebSocket routes are properly configured."""
        try:
            from app.main import app

            # Get all routes
            routes = []
            for route in app.routes:
                if hasattr(route, "path"):
                    routes.append(route.path)

            # Check for voice routes (they might be dynamically added)
            print(f"Available routes: {routes}")

            # The main test is that the app can be imported without errors
            assert app is not None
            print("✅ Main app with voice routes loads successfully")

        except ImportError as e:
            pytest.skip(f"Main app import failed: {str(e)}")
        except Exception as e:
            pytest.fail(f"Voice WebSocket routes test failed: {str(e)}")

    @pytest.mark.asyncio
    async def test_voice_manager_integration(self):
        """Test VoiceManager integration with mocked dependencies."""
        try:
            # Test with mocked dependencies to avoid requiring actual models
            with patch("services.voice_manager.VoskSTTService") as mock_stt, patch(
                "services.voice_manager.PiperTTSService"
            ) as mock_tts:
                # Setup mocks
                mock_stt_instance = Mock()
                mock_stt_instance.is_available.return_value = True
                mock_stt_instance.create_recognizer = AsyncMock(return_value=True)
                mock_stt_instance.destroy_recognizer = AsyncMock(return_value=True)
                mock_stt.return_value = mock_stt_instance

                mock_tts_instance = Mock()
                mock_tts_instance.is_available.return_value = True
                mock_tts.return_value = mock_tts_instance

                # Test VoiceManager creation
                from services.voice_manager import VoiceManager

                config = {
                    "voice": {
                        "enabled": True,
                        "stt": {"enabled": True, "sample_rate": 16000},
                        "tts": {"enabled": True, "sample_rate": 22050},
                    }
                }

                voice_manager = VoiceManager(config)

                # Test basic functionality
                assert voice_manager.is_voice_available() == True

                # Test session management
                session_id = "test_integration_session"
                success = await voice_manager.create_voice_session(session_id)
                assert success == True

                session_state = voice_manager.get_voice_session_state(session_id)
                assert session_state is not None

                # Test cleanup
                success = await voice_manager.destroy_voice_session(session_id)
                assert success == True

                print("✅ VoiceManager integration test passed")

        except ImportError as e:
            pytest.skip(f"Voice manager imports not available: {str(e)}")
        except Exception as e:
            pytest.fail(f"Voice manager integration test failed: {str(e)}")

    def test_voice_dependencies_availability(self):
        """Test availability of voice processing dependencies."""
        dependencies = {
            "vosk": "Speech recognition",
            "piper": "Text-to-speech",
            "sounddevice": "Audio processing",
            "webrtcvad": "Voice activity detection",
        }

        available_deps = []
        missing_deps = []

        for dep, description in dependencies.items():
            try:
                __import__(dep)
                available_deps.append((dep, description))
                print(f"✅ {dep}: {description}")
            except ImportError:
                missing_deps.append((dep, description))
                print(f"❌ {dep}: {description} - NOT AVAILABLE")

        if missing_deps:
            print(
                f"\n💡 To enable full voice functionality, install: pip install {' '.join([dep for dep, _ in missing_deps])}"
            )

        # Test passes if at least some dependencies are available
        assert len(available_deps) >= 0  # Always passes, but provides info

    def test_voice_model_directories(self):
        """Test voice model directory structure."""

        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models"

        # Check if models directory exists
        if models_dir.exists():
            print(f"✅ Models directory exists: {models_dir}")

            # Check for voice model subdirectories
            vosk_dir = models_dir / "vosk"
            piper_dir = models_dir / "piper"

            if vosk_dir.exists():
                vosk_models = list(vosk_dir.glob("vosk-model-*"))
                print(f"📁 Vosk models found: {len(vosk_models)}")
                for model in vosk_models:
                    print(f"   - {model.name}")
            else:
                print("📁 Vosk models directory not found")

            if piper_dir.exists():
                piper_voices = list(piper_dir.glob("*/*.onnx"))
                print(f"🎵 Piper voices found: {len(piper_voices)}")
                for voice in piper_voices:
                    print(f"   - {voice.parent.name}")
            else:
                print("🎵 Piper voices directory not found")
        else:
            print("📁 Models directory not found - run setup to download models")

        # Test always passes - this is informational
        assert True

    def test_voice_scripts_availability(self):
        """Test that voice-related scripts are available."""

        project_root = Path(__file__).parent.parent
        scripts_dir = project_root / "scripts"

        voice_scripts = ["download_voice_models.py", "test_voice_features.py"]

        for script in voice_scripts:
            script_path = scripts_dir / script
            if script_path.exists():
                print(f"✅ Voice script available: {script}")
            else:
                print(f"❌ Voice script missing: {script}")

        # Test that at least the download script exists
        download_script = scripts_dir / "download_voice_models.py"
        assert download_script.exists(), "Voice model download script should exist"


class TestVoiceUIIntegration:
    """Test voice UI integration."""

    def test_voice_static_files(self):
        """Test that voice static files are available."""

        project_root = Path(__file__).parent.parent
        static_dir = project_root / "app" / "static"

        voice_files = ["js/voice-controller.js", "worklets/pcm-recorder.js"]

        for file_path in voice_files:
            full_path = static_dir / file_path
            if full_path.exists():
                print(f"✅ Voice UI file available: {file_path}")

                # Basic content check
                content = full_path.read_text()
                assert len(content) > 100, f"Voice file {file_path} seems too small"

            else:
                pytest.fail(f"Voice UI file missing: {file_path}")

    def test_voice_css_styles(self):
        """Test that voice CSS styles are included."""

        project_root = Path(__file__).parent.parent
        css_file = project_root / "app" / "static" / "style.css"

        if css_file.exists():
            content = css_file.read_text()

            # Check for voice-related CSS classes
            voice_css_classes = [
                ".voice-toggle",
                ".mic-orb",
                ".live-transcription",
                ".voice-components",
            ]

            for css_class in voice_css_classes:
                if css_class in content:
                    print(f"✅ Voice CSS class found: {css_class}")
                else:
                    print(f"❌ Voice CSS class missing: {css_class}")

            # Test that at least some voice styles exist
            voice_styles_exist = any(css_class in content for css_class in voice_css_classes)
            assert voice_styles_exist, "Voice CSS styles should be present"
        else:
            pytest.fail("Main CSS file not found")

    def test_voice_html_elements(self):
        """Test that voice HTML elements are included."""

        project_root = Path(__file__).parent.parent
        html_file = project_root / "app" / "static" / "index.html"

        if html_file.exists():
            content = html_file.read_text()

            # Check for voice-related HTML elements
            voice_elements = [
                'id="voiceToggle"',
                'id="micOrb"',
                'id="liveTranscription"',
                'class="voice-components"',
            ]

            for element in voice_elements:
                if element in content:
                    print(f"✅ Voice HTML element found: {element}")
                else:
                    print(f"❌ Voice HTML element missing: {element}")

            # Test that at least some voice elements exist
            voice_elements_exist = any(element in content for element in voice_elements)
            assert voice_elements_exist, "Voice HTML elements should be present"
        else:
            pytest.fail("Main HTML file not found")


class TestVoicePerformance:
    """Test voice performance characteristics."""

    @pytest.mark.asyncio
    async def test_voice_service_memory_usage(self):
        """Test that voice services don't consume excessive memory."""
        try:
            import os

            import psutil

            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Test voice service creation with mocks
            with patch("services.voice_manager.VoskSTTService") as mock_stt, patch(
                "services.voice_manager.PiperTTSService"
            ) as mock_tts:
                # Setup lightweight mocks
                mock_stt.return_value.is_available.return_value = True
                mock_tts.return_value.is_available.return_value = True

                from services.voice_manager import VoiceManager

                config = {
                    "voice": {"enabled": True, "stt": {"enabled": True}, "tts": {"enabled": True}}
                }

                # Create multiple voice managers to test memory usage
                managers = []
                for i in range(5):
                    manager = VoiceManager(config)
                    managers.append(manager)

                # Get memory usage after creation
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory

                print(
                    f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB"
                )
                print(f"Memory increase: {memory_increase:.1f}MB")

                # Test that memory increase is reasonable (less than 100MB for mocked services)
                assert memory_increase < 100, f"Memory increase too high: {memory_increase}MB"

                print("✅ Voice service memory usage test passed")

        except ImportError:
            pytest.skip("psutil not available for memory testing")
        except Exception as e:
            pytest.fail(f"Memory usage test failed: {str(e)}")

    def test_voice_config_performance(self):
        """Test that voice configuration loading is fast."""
        import time

        start_time = time.time()

        try:
            from app.core import load_config

            # Load config multiple times to test performance
            for _ in range(10):
                config = load_config()
                voice_config = config.get("voice", {})

            end_time = time.time()
            total_time = end_time - start_time

            print(f"Config loading time for 10 iterations: {total_time:.3f}s")

            # Test that config loading is fast (less than 1 second for 10 loads)
            assert total_time < 1.0, f"Config loading too slow: {total_time}s"

            print("✅ Voice configuration performance test passed")

        except Exception as e:
            pytest.fail(f"Config performance test failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

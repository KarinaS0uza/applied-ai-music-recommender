"""Wire concrete providers into the discovery workflow for the CLI and Streamlit app."""

from __future__ import annotations

import logging
from dataclasses import replace

from src.discovery import discover_music
from src.knowledge import retrieve_documents
from src.models import DiscoveryError, DiscoveryResult, MusicRequest
from src.providers import (GeminiProvider, MusicBrainzProvider, Settings, TavilyProvider,
                           YouTubeProvider)
from src.validation import validate_request

logger = logging.getLogger(__name__)


def attach_videos(result: DiscoveryResult, youtube: YouTubeProvider) -> DiscoveryResult:
    """Give each recommendation its exact YouTube video; songs without a confident match are untouched.

    Videos are a bonus on top of verified results, so no failure here may cost the user their results.
    """
    try:
        videos = youtube.find_videos([(rec.title, rec.artist) for rec in result.recommendations])
        result.recommendations = [replace(rec, video=video) if video else rec
                                  for rec, video in zip(result.recommendations, videos)]
    except Exception:
        logger.warning("youtube_lookup_failed code=unexpected")
    return result


def find_music(request: MusicRequest, settings: Settings | None = None) -> DiscoveryResult:
    """Configure providers and run the shared CLI and Streamlit workflow."""
    validate_request(request)
    settings = settings or Settings.from_mapping()
    gemini = GeminiProvider(settings)
    web = TavilyProvider(settings.tavily_api_key)
    youtube = YouTubeProvider(settings.youtube_api_key) if settings.youtube_api_key else None
    mb = None
    try:
        documents = metadata = None
        if request.enhanced:
            documents = retrieve_documents
            mb = MusicBrainzProvider(settings.musicbrainz_contact)
            metadata = mb.lookup_recordings
        result = discover_music(request, analyze=gemini.analyze_request, search=web.search_web,
                                extract=gemini.extract_candidates, generate=gemini.generate_recommendations,
                                documents=documents, metadata=metadata, retry_invalid=True,
                                degrade_invalid=True)
        return attach_videos(result, youtube) if youtube and result.recommendations else result
    except DiscoveryError:
        raise
    except Exception:
        logger.warning("discovery_failed code=unexpected")
        raise DiscoveryError("provider_unavailable", "Discovery could not finish. Please try again later.") from None
    finally:
        sessions = [web.session, *(p.session for p in (mb, youtube) if p)]
        for close in [gemini.close, *(session.close for session in sessions)]:
            try:
                close()
            except Exception:
                logger.info("provider_cleanup_failed")

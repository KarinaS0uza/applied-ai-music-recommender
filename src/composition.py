"""Wire concrete providers into the discovery workflow for the CLI and Streamlit app."""

from __future__ import annotations

import logging
from dataclasses import replace

from src.discovery import discover_music
from src.knowledge import retrieve_documents
from src.models import DiscoveryError, DiscoveryResult, MusicRequest
from src.providers import (GeminiEmbeddingProvider, GeminiProvider, MusicBrainzProvider,
                           Settings, SpotifyProvider, TavilyProvider, YouTubeProvider)
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


def attach_tracks(result: DiscoveryResult, spotify: SpotifyProvider) -> DiscoveryResult:
    """Give each recommendation its exact Spotify track; songs without a confident match are untouched.

    Like videos, tracks are a bonus on top of verified results, so no failure here may cost the user their results.
    """
    try:
        tracks = spotify.find_tracks([(rec.title, rec.artist) for rec in result.recommendations])
        result.recommendations = [replace(rec, track=track) if track else rec
                                  for rec, track in zip(result.recommendations, tracks)]
    except Exception:
        logger.warning("spotify_lookup_failed code=unexpected")
    return result


def find_music(request: MusicRequest, settings: Settings | None = None) -> DiscoveryResult:
    """Configure providers and run the shared CLI and Streamlit workflow."""
    validate_request(request)
    settings = settings or Settings.from_mapping()
    gemini = GeminiProvider(settings)
    web = TavilyProvider(settings.tavily_api_key,
                         music_source_focus_enabled=settings.music_source_focus_enabled)
    youtube = YouTubeProvider(settings.youtube_api_key) if settings.youtube_api_key else None
    spotify = (SpotifyProvider(settings.spotify_client_id, settings.spotify_client_secret)
               if settings.spotify_client_id and settings.spotify_client_secret else None)
    mb = None
    try:
        documents = metadata = None
        if request.enhanced:
            if settings.semantic_retrieval_enabled:
                embedding = GeminiEmbeddingProvider(settings, client=gemini.client)
                documents = lambda current_request, intent: retrieve_documents(
                    current_request, intent, semantic_ranker=embedding.score)
            else:
                documents = retrieve_documents
            mb = MusicBrainzProvider(settings.musicbrainz_contact)
            metadata = mb.lookup_recordings
        result = discover_music(request, analyze=gemini.analyze_request, search=web.search_web,
                                extract=gemini.extract_candidates, generate=gemini.generate_recommendations,
                                documents=documents, metadata=metadata, retry_invalid=True,
                                degrade_invalid=True)
        if result.recommendations:
            if youtube:
                result = attach_videos(result, youtube)
            if spotify:
                result = attach_tracks(result, spotify)
        return result
    except DiscoveryError:
        raise
    except Exception:
        logger.warning("discovery_failed code=unexpected")
        raise DiscoveryError("provider_unavailable", "Discovery could not finish. Please try again later.") from None
    finally:
        sessions = [web.session, *(p.session for p in (mb, youtube, spotify) if p)]
        for close in [gemini.close, *(session.close for session in sessions)]:
            try:
                close()
            except Exception:
                logger.info("provider_cleanup_failed")

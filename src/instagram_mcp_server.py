#!/usr/bin/env python3
"""
Instagram MCP Server - A Model Context Protocol server for Instagram API integration.

This server provides tools, resources, and prompts for interacting with Instagram's Graph API,
enabling AI applications to manage Instagram business accounts programmatically.

HTTP transport + auth (``_run_http``) is adapted from the mbay-ODW signal-mcp /
whatsapp-mcp servers: Streamable-HTTP + classic SSE, a static ``MCP_API_KEY``
bearer and Authelia OIDC introspection, plus the RFC 9728/8414 OAuth discovery
endpoints Claude.ai needs.
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import structlog
from mcp.server import Server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Prompt, Resource, TextContent, Tool

from .config import get_settings
from .instagram_client import InstagramAPIError, InstagramClient
from .models.instagram_models import (
    InsightMetric,
    InsightPeriod,
    MCPToolResult,
    PublishMediaRequest,
)

# Configure logging
logger = structlog.get_logger(__name__)

# Global Instagram client
instagram_client: Optional[InstagramClient] = None


class InstagramMCPServer:
    """Instagram MCP Server implementation."""

    def __init__(self):
        self.settings = get_settings()
        self.server = Server(self.settings.mcp_server_name)
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP server handlers."""

        # Tools
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="get_profile_info",
                    description=(
                        "Get Instagram business profile information including "
                        "followers, bio, and account details"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "account_id": {
                                "type": "string",
                                "description": (
                                    "Instagram business account ID (optional, "
                                    "uses configured account if not provided)"
                                ),
                            }
                        },
                    },
                ),
                Tool(
                    name="get_media_posts",
                    description=(
                        "Get recent media posts from Instagram account "
                        "with engagement metrics"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "account_id": {
                                "type": "string",
                                "description": "Instagram business account ID (optional)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of posts to retrieve (max 100)",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 25,
                            },
                            "after": {
                                "type": "string",
                                "description": (
                                    "Pagination cursor for getting posts "
                                    "after a specific point"
                                ),
                            },
                        },
                    },
                ),
                Tool(
                    name="get_media_insights",
                    description=(
                        "Get detailed insights and analytics for a "
                        "specific Instagram post"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "media_id": {
                                "type": "string",
                                "description": "Instagram media ID to get insights for",
                            },
                            "metrics": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "reach",
                                        "likes",
                                        "comments",
                                        "shares",
                                        "saved",
                                        "video_views",
                                    ],
                                },
                                "description": (
                                    "Specific metrics to retrieve (optional, "
                                    "gets all available if not specified). "
                                    "Note: video_views only works for video posts"
                                ),
                            },
                        },
                        "required": ["media_id"],
                    },
                ),
                Tool(
                    name="publish_media",
                    description=(
                        "Upload and publish an image or video to Instagram "
                        "with caption and optional location"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_url": {
                                "type": "string",
                                "format": "uri",
                                "description": (
                                    "URL of the image to publish "
                                    "(must be publicly accessible)"
                                ),
                            },
                            "video_url": {
                                "type": "string",
                                "format": "uri",
                                "description": (
                                    "URL of the video to publish "
                                    "(must be publicly accessible)"
                                ),
                            },
                            "caption": {
                                "type": "string",
                                "description": "Caption for the post (optional)",
                            },
                            "location_id": {
                                "type": "string",
                                "description": (
                                    "Facebook location ID for geotagging (optional)"
                                ),
                            },
                        },
                        "anyOf": [
                            {"required": ["image_url"]},
                            {"required": ["video_url"]},
                        ],
                    },
                ),
                Tool(
                    name="get_account_pages",
                    description=(
                        "Get Facebook pages connected to the account and "
                        "their Instagram business accounts"
                    ),
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="get_account_insights",
                    description=(
                        "Get account-level insights and analytics for "
                        "Instagram business account"
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "account_id": {
                                "type": "string",
                                "description": "Instagram business account ID (optional)",
                            },
                            "metrics": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "reach",
                                        "profile_views",
                                        "website_clicks",
                                        "accounts_engaged",
                                    ],
                                },
                                "description": "Specific metrics to retrieve (Note: follower_count is available via get_profile_info)",
                            },
                            "period": {
                                "type": "string",
                                "enum": ["day", "lifetime"],
                                "description": "Time period for insights (day for engagement metrics, lifetime for demographics)",
                                "default": "day",
                            },
                        },
                    },
                ),
                Tool(
                    name="validate_access_token",
                    description=(
                        "Validate the Instagram API access token and "
                        "check permissions"
                    ),
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="get_conversations",
                    description=(
                        "Get Instagram DM conversations. "
                        "Requires instagram_manage_messages permission. "
                        "Lists all conversations for the connected Instagram account."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "page_id": {
                                "type": "string",
                                "description": (
                                    "Facebook page ID (optional, auto-detected from "
                                    "connected pages if not provided)"
                                ),
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of conversations to retrieve (max 100)",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 25,
                            },
                        },
                    },
                ),
                Tool(
                    name="get_conversation_messages",
                    description=(
                        "Get messages from a specific Instagram DM conversation. "
                        "Requires instagram_manage_messages permission. "
                        "Use get_conversations to get conversation IDs."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "conversation_id": {
                                "type": "string",
                                "description": "Instagram conversation ID",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of messages to retrieve (max 100)",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 25,
                            },
                        },
                        "required": ["conversation_id"],
                    },
                ),
                Tool(
                    name="send_dm",
                    description=(
                        "Send Instagram direct message to a user. "
                        "IMPORTANT: Requires instagram_manage_messages with Advanced Access from Meta. "
                        "Can only reply within 24 hours of user's last message. "
                        "Recipient must have initiated conversation first."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "recipient_id": {
                                "type": "string",
                                "description": "Instagram Scoped User ID (IGSID) of recipient",
                            },
                            "message": {
                                "type": "string",
                                "description": "Message text to send (max 1000 characters)",
                                "maxLength": 1000,
                            },
                        },
                        "required": ["recipient_id", "message"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> Sequence[TextContent]:
            """Handle tool calls."""
            global instagram_client

            if not instagram_client:
                instagram_client = InstagramClient()

            try:
                if name == "get_profile_info":
                    account_id = arguments.get("account_id")
                    profile = await instagram_client.get_profile_info(account_id)

                    result = MCPToolResult(
                        success=True,
                        data=profile.model_dump(mode='json'),
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif name == "get_media_posts":
                    account_id = arguments.get("account_id")
                    limit = arguments.get("limit", 25)
                    after = arguments.get("after")

                    posts = await instagram_client.get_media_posts(
                        account_id, limit, after
                    )

                    result = MCPToolResult(
                        success=True,
                        data={
                            "posts": [post.model_dump(mode='json') for post in posts],
                            "count": len(posts),
                        },
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif name == "get_media_insights":
                    media_id = arguments["media_id"]
                    metrics = arguments.get("metrics")

                    if metrics:
                        metrics = [InsightMetric(m) for m in metrics]

                    insights = await instagram_client.get_media_insights(
                        media_id, metrics
                    )

                    result = MCPToolResult(
                        success=True,
                        data={
                            "media_id": media_id,
                            "insights": [insight.model_dump(mode='json') for insight in insights],
                        },
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif name == "publish_media":
                    request = PublishMediaRequest(**arguments)
                    response = await instagram_client.publish_media(request)

                    result = MCPToolResult(
                        success=True,
                        data=response.model_dump(mode='json'),
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif name == "get_account_pages":
                    pages = await instagram_client.get_account_pages()

                    result = MCPToolResult(
                        success=True,
                        data={
                            "pages": [page.model_dump(mode='json') for page in pages],
                            "count": len(pages),
                        },
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif name == "get_account_insights":
                    account_id = arguments.get("account_id")
                    metrics = arguments.get("metrics")
                    period = InsightPeriod(arguments.get("period", "day"))

                    insights = await instagram_client.get_account_insights(
                        account_id, metrics, period
                    )

                    result = MCPToolResult(
                        success=True,
                        data={
                            "insights": [insight.model_dump(mode='json') for insight in insights],
                            "period": period.value,
                        },
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif name == "validate_access_token":
                    is_valid = await instagram_client.validate_access_token()

                    result = MCPToolResult(
                        success=True,
                        data={"valid": is_valid},
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif name == "get_conversations":
                    page_id = arguments.get("page_id")
                    limit = arguments.get("limit", 25)

                    conversations = await instagram_client.get_conversations(
                        page_id, limit
                    )

                    result = MCPToolResult(
                        success=True,
                        data={
                            "conversations": [conv.model_dump(mode='json') for conv in conversations],
                            "count": len(conversations),
                        },
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                            "note": "Requires instagram_manage_messages permission"
                        },
                    )

                elif name == "get_conversation_messages":
                    conversation_id = arguments["conversation_id"]
                    limit = arguments.get("limit", 25)

                    messages = await instagram_client.get_conversation_messages(
                        conversation_id, limit
                    )

                    result = MCPToolResult(
                        success=True,
                        data={
                            "conversation_id": conversation_id,
                            "messages": [msg.model_dump(mode='json') for msg in messages],
                            "count": len(messages),
                        },
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

                elif name == "send_dm":
                    from .models.instagram_models import SendDMRequest

                    request = SendDMRequest(**arguments)
                    response = await instagram_client.send_dm(request)

                    result = MCPToolResult(
                        success=True,
                        data=response.model_dump(mode='json'),
                        metadata={
                            "tool": name,
                            "timestamp": datetime.utcnow().isoformat(),
                            "note": "24-hour response window applies. Requires Advanced Access."
                        },
                    )

                else:
                    result = MCPToolResult(success=False, error=f"Unknown tool: {name}")

            except InstagramAPIError as e:
                logger.error("Instagram API error", tool=name, error=str(e))
                result = MCPToolResult(
                    success=False,
                    error=f"Instagram API error: {e.message}",
                    metadata={
                        "error_code": e.error_code,
                        "error_subcode": e.error_subcode,
                    },
                )

            except Exception as e:
                logger.error("Tool execution error", tool=name, error=str(e))
                result = MCPToolResult(
                    success=False, error=f"Tool execution failed: {str(e)}"
                )

            return [TextContent(type="text", text=json.dumps(result.model_dump(mode='json'), indent=2))]

        # Resources
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="instagram://profile",
                    name="Instagram Profile",
                    description="Current Instagram business profile information",
                    mimeType="application/json",
                ),
                Resource(
                    uri="instagram://media/recent",
                    name="Recent Media Posts",
                    description="Recent Instagram posts with engagement metrics",
                    mimeType="application/json",
                ),
                Resource(
                    uri="instagram://insights/account",
                    name="Account Insights",
                    description="Account-level analytics and insights",
                    mimeType="application/json",
                ),
                Resource(
                    uri="instagram://pages",
                    name="Connected Pages",
                    description="Facebook pages connected to the account",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Handle resource reading."""
            global instagram_client

            if not instagram_client:
                instagram_client = InstagramClient()

            try:
                if uri == "instagram://profile":
                    profile = await instagram_client.get_profile_info()
                    return json.dumps(profile.model_dump(mode='json'), indent=2)

                elif uri == "instagram://media/recent":
                    posts = await instagram_client.get_media_posts(limit=10)
                    return json.dumps([post.model_dump(mode='json') for post in posts], indent=2)

                elif uri == "instagram://insights/account":
                    insights = await instagram_client.get_account_insights()
                    return json.dumps(
                        [insight.model_dump(mode='json') for insight in insights], indent=2
                    )

                elif uri == "instagram://pages":
                    pages = await instagram_client.get_account_pages()
                    return json.dumps([page.model_dump(mode='json') for page in pages], indent=2)

                else:
                    raise ValueError(f"Unknown resource URI: {uri}")

            except Exception as e:
                logger.error("Resource read error", uri=uri, error=str(e))
                return json.dumps({"error": str(e)}, indent=2)

        # Prompts
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[Prompt]:
            """List available prompts."""
            return [
                Prompt(
                    name="analyze_engagement",
                    description="Analyze Instagram post engagement and provide insights",
                    arguments=[
                        {
                            "name": "media_id",
                            "description": "Instagram media ID to analyze",
                            "required": True,
                        },
                        {
                            "name": "comparison_period",
                            "description": "Period to compare against (e.g., 'last_week', 'last_month')",
                            "required": False,
                        },
                    ],
                ),
                Prompt(
                    name="content_strategy",
                    description="Generate content strategy recommendations based on account performance",
                    arguments=[
                        {
                            "name": "focus_area",
                            "description": "Area to focus on (e.g., 'engagement', 'reach', 'growth')",
                            "required": False,
                        },
                        {
                            "name": "time_period",
                            "description": "Time period to analyze (e.g., 'week', 'month')",
                            "required": False,
                        },
                    ],
                ),
                Prompt(
                    name="hashtag_analysis",
                    description="Analyze hashtag performance and suggest improvements",
                    arguments=[
                        {
                            "name": "post_count",
                            "description": "Number of recent posts to analyze",
                            "required": False,
                        }
                    ],
                ),
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: Dict[str, str]) -> str:
            """Handle prompt requests."""
            global instagram_client

            if not instagram_client:
                instagram_client = InstagramClient()

            try:
                if name == "analyze_engagement":
                    media_id = arguments.get("media_id")
                    if not media_id:
                        return "Error: media_id is required for engagement analysis"

                    # Get media insights
                    insights = await instagram_client.get_media_insights(media_id)

                    prompt = f"""
Analyze the engagement metrics for Instagram post {media_id}:

Insights Data:
{json.dumps([insight.model_dump(mode='json') for insight in insights], indent=2)}

Please provide:
1. Overall engagement performance assessment
2. Key metrics analysis (impressions, reach, likes, comments, shares)
3. Engagement rate calculation and interpretation
4. Recommendations for improving future posts
5. Comparison with typical performance benchmarks
"""
                    return prompt

                elif name == "content_strategy":
                    focus_area = arguments.get("focus_area", "engagement")
                    time_period = arguments.get("time_period", "week")

                    # Get recent posts and account insights
                    posts = await instagram_client.get_media_posts(limit=20)
                    account_insights = await instagram_client.get_account_insights()

                    prompt = f"""
Generate a content strategy for Instagram focusing on {focus_area} over the {time_period}:

Recent Posts Performance:
{json.dumps([post.model_dump(mode='json') for post in posts[:5]], indent=2)}

Account Insights:
{json.dumps([insight.model_dump(mode='json') for insight in account_insights], indent=2)}

Please provide:
1. Content performance analysis
2. Optimal posting times and frequency
3. Content type recommendations (images, videos, carousels)
4. Caption and hashtag strategies
5. Engagement tactics to improve {focus_area}
6. Specific action items for the next {time_period}
"""
                    return prompt

                elif name == "hashtag_analysis":
                    post_count = int(arguments.get("post_count", "10"))

                    # Get recent posts
                    posts = await instagram_client.get_media_posts(limit=post_count)

                    # Extract hashtags from captions
                    hashtags_data = []
                    for post in posts:
                        if post.caption:
                            hashtags = [
                                word
                                for word in post.caption.split()
                                if word.startswith("#")
                            ]
                            hashtags_data.append(
                                {
                                    "post_id": post.id,
                                    "hashtags": hashtags,
                                    "likes": post.like_count,
                                    "comments": post.comments_count,
                                }
                            )

                    prompt = f"""
Analyze hashtag performance for the last {post_count} Instagram posts:

Hashtag Data:
{json.dumps(hashtags_data, indent=2)}

Please provide:
1. Most frequently used hashtags
2. Hashtag performance correlation with engagement
3. Hashtag diversity analysis
4. Recommendations for hashtag optimization
5. Suggested new hashtags to try
6. Hashtag strategy improvements
"""
                    return prompt

                else:
                    return f"Error: Unknown prompt '{name}'"

            except Exception as e:
                logger.error("Prompt generation error", prompt=name, error=str(e))
                return f"Error generating prompt: {str(e)}"

    async def run(self):
        """Run the MCP server."""
        logger.info(
            "Starting Instagram MCP Server", version=self.settings.mcp_server_version
        )

        # Initialize Instagram client
        global instagram_client
        instagram_client = InstagramClient()

        # Validate access token on startup
        try:
            is_valid = await instagram_client.validate_access_token()
            if not is_valid:
                logger.error("Invalid Instagram access token")
                sys.exit(1)
            logger.info("Instagram access token validated successfully")
        except Exception as e:
            logger.error("Failed to validate access token", error=str(e))
            sys.exit(1)

        if self.settings.mcp_transport == "sse":
            self._run_http()
            return

        # Run the server (stdio)
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=self.settings.mcp_server_name,
                    server_version=self.settings.mcp_server_version,
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    ),
                ),
            )

    # --------------------------------------------------------------------------
    # HTTP transport + auth (adapted from signal-mcp / whatsapp-mcp)
    # --------------------------------------------------------------------------
    def _run_http(self) -> None:
        """Serve over Streamable-HTTP + classic SSE, guarded by MCP_API_KEY /
        Authelia OIDC token introspection, with the OAuth discovery endpoints
        Claude.ai needs to complete its connector auth flow."""
        import contextlib
        import logging
        import os
        import time
        from collections.abc import AsyncIterator

        import httpx as _httpx
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Mount, Route

        realm = "instagram-mcp"
        _server = self.server

        _log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        _level_int = getattr(logging, _log_level, logging.INFO)
        logging.basicConfig(
            level=_level_int,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            force=True,
        )
        logging.getLogger().setLevel(_level_int)
        log = logging.getLogger(realm)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(name).setLevel(_level_int)

        mcp_api_key = os.getenv("MCP_API_KEY", "")
        oidc_introspection_url = os.getenv("OIDC_INTROSPECTION_URL", "")
        oidc_client_id = os.getenv("OIDC_CLIENT_ID", "")
        oidc_client_secret = os.getenv("OIDC_CLIENT_SECRET", "")
        oauth_issuer = os.getenv("OAUTH_ISSUER", "")
        mcp_server_url = os.getenv("MCP_SERVER_URL", "")

        auth_configured = bool(mcp_api_key) or all(
            (oidc_introspection_url, oidc_client_id, oidc_client_secret)
        )
        if not auth_configured:
            log.warning(
                "[auth] NEITHER MCP_API_KEY NOR a complete OIDC triple configured — "
                "ALL requests pass unauthenticated."
            )

        def _auth_preview(auth: str) -> str:
            return "(none)" if not auth else auth[:20] + ("…" if len(auth) > 20 else "")

        async def _is_authorized(request: Request) -> tuple[bool, Optional[str]]:
            tag = f"{request.method} {request.url.path}"
            auth = request.headers.get("Authorization", "")
            if not mcp_api_key and not (
                oidc_introspection_url and oidc_client_id and oidc_client_secret
            ):
                return True, None
            if not auth:
                log.warning("[auth] %s — DENY: no Authorization header", tag)
                return False, "no_header"
            if mcp_api_key and auth == f"Bearer {mcp_api_key}":
                log.info("[auth] %s — OK: static MCP_API_KEY", tag)
                return True, None
            if not auth.startswith("Bearer "):
                return False, "invalid_token"
            if not (oidc_introspection_url and oidc_client_id and oidc_client_secret):
                return False, "invalid_token"
            jwt_token = auth[7:]
            try:
                async with _httpx.AsyncClient(timeout=5.0) as http:
                    resp = await http.post(
                        oidc_introspection_url,
                        data={"token": jwt_token},
                        auth=(oidc_client_id, oidc_client_secret),
                    )
                    if resp.status_code != 200:
                        log.warning(
                            "[auth] %s — DENY: introspection HTTP %s", tag, resp.status_code
                        )
                        return False, "invalid_token"
                    if bool(resp.json().get("active")):
                        log.info("[auth] %s — OK: OIDC token active", tag)
                        return True, None
                    return False, "invalid_token"
            except Exception as e:
                log.error("[auth] %s — introspection error: %s", tag, e)
                return False, "invalid_token"

        def _unauthorized(reason: Optional[str]) -> Response:
            if reason == "invalid_token":
                www = (
                    f'Bearer realm="{realm}", error="invalid_token", '
                    'error_description="The access token expired or is invalid"'
                )
                return Response(
                    "Unauthorized", status_code=401, headers={"WWW-Authenticate": www}
                )
            return Response("Unauthorized", status_code=401)

        class RequestLogMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):  # type: ignore[override]
                t0 = time.monotonic()
                log.debug(
                    "→ %s %s auth=%s",
                    request.method,
                    request.url.path,
                    _auth_preview(request.headers.get("Authorization", "")),
                )
                response = await call_next(request)
                log.debug(
                    "← %s %s → %d in %dms",
                    request.method,
                    request.url.path,
                    response.status_code,
                    int((time.monotonic() - t0) * 1000),
                )
                return response

        sse = SseServerTransport("/messages/")
        session_manager = StreamableHTTPSessionManager(app=_server, json_response=True)

        class _AlreadySent(Response):
            def __init__(self) -> None:
                super().__init__(content=b"", status_code=200)

            async def __call__(self, scope, receive, send):
                return

        async def handle_streamable_http(request: Request):
            ok, reason = await _is_authorized(request)
            if not ok:
                return _unauthorized(reason)
            await session_manager.handle_request(request.scope, request.receive, request._send)
            return _AlreadySent()

        @contextlib.asynccontextmanager
        async def lifespan(_app: "Starlette") -> AsyncIterator[None]:
            async with session_manager.run():
                log.info("StreamableHTTPSessionManager started")
                yield

        async def handle_sse(request: Request):
            ok, reason = await _is_authorized(request)
            if not ok:
                return _unauthorized(reason)
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await _server.run(streams[0], streams[1], _server.create_initialization_options())
            return Response()

        async def handle_messages(scope, receive, send):
            req = Request(scope, receive=receive)
            ok, reason = await _is_authorized(req)
            if not ok:
                await _unauthorized(reason)(scope, receive, send)
                return
            await sse.handle_post_message(scope, receive, send)

        async def handle_oauth_protected_resource(request: Request):
            return JSONResponse({
                "resource": mcp_server_url or str(request.base_url).rstrip("/"),
                "authorization_servers": [oauth_issuer] if oauth_issuer else [],
                "bearer_methods_supported": ["header"],
                "scopes_supported": ["openid", "profile", "email"],
            })

        async def handle_oauth_authorization_server(request: Request):
            if oauth_issuer:
                try:
                    async with _httpx.AsyncClient(timeout=5.0) as http:
                        up = await http.get(f"{oauth_issuer}/.well-known/oauth-authorization-server")
                        if up.status_code == 200:
                            return JSONResponse(up.json())
                except Exception as e:
                    log.warning("[discovery] upstream fetch failed: %s", e)
            return JSONResponse({
                "issuer": oauth_issuer,
                "authorization_endpoint": f"{oauth_issuer}/api/oidc/authorization",
                "token_endpoint": f"{oauth_issuer}/api/oidc/token",
                "jwks_uri": f"{oauth_issuer}/jwks.json",
                "introspection_endpoint": f"{oauth_issuer}/api/oidc/introspection",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
                "scopes_supported": ["openid", "profile", "email"],
            })

        app = Starlette(
            routes=[
                Route("/.well-known/oauth-protected-resource",
                      endpoint=handle_oauth_protected_resource, methods=["GET"]),
                Route("/.well-known/oauth-authorization-server",
                      endpoint=handle_oauth_authorization_server, methods=["GET"]),
                Route("/sse", endpoint=handle_streamable_http, methods=["POST"]),
                Route("/mcp", endpoint=handle_streamable_http, methods=["POST"]),
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=handle_messages),
            ],
            middleware=[Middleware(RequestLogMiddleware)],
            lifespan=lifespan,
        )

        port = int(os.getenv("PORT", "8000"))
        log.info("%s listening on :%d (LOG_LEVEL=%s)", realm, port, _log_level)
        uvicorn.run(app, host="0.0.0.0", port=port)


async def main():
    """Main entry point."""
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set log level
    import logging

    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level))

    # Create and run server
    server = InstagramMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())

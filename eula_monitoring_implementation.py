"""
Real-time Software EULA Analysis - Technical Implementation Example
This module demonstrates the core components of an agentic EULA monitoring system.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
from bs4 import BeautifulSoup
import difflib
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import schedule
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SoftwareAsset:
    """Represents a software asset being monitored."""
    name: str
    vendor: str
    version: str
    eula_url: str
    last_checked: datetime
    current_hash: str
    license_type: str
    business_critical: bool = False

@dataclass
class EULAChange:
    """Represents a detected change in a EULA."""
    software: SoftwareAsset
    old_content: str
    new_content: str
    detected_at: datetime
    change_type: str
    risk_level: RiskLevel
    impact_summary: str
    recommended_actions: List[str]

class EULAMonitor:
    """Core EULA monitoring service."""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            openai_api_key=openai_api_key,
            temperature=0.1
        )
        self.software_assets: Dict[str, SoftwareAsset] = {}
        self.notification_handlers = []
        
    def register_software(self, asset: SoftwareAsset):
        """Register a software asset for monitoring."""
        self.software_assets[asset.name] = asset
        logger.info(f"Registered software: {asset.name} by {asset.vendor}")
        
    def add_notification_handler(self, handler):
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
        
    async def fetch_eula_content(self, url: str) -> Optional[str]:
        """Fetch EULA content from URL with error handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove common non-content elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
                
            text = soup.get_text(separator=' ', strip=True)
            return text
            
        except Exception as e:
            logger.error(f"Failed to fetch EULA from {url}: {e}")
            return None
            
    def calculate_content_hash(self, content: str) -> str:
        """Calculate hash of normalized content for change detection."""
        # Normalize content by removing extra whitespace and common variations
        normalized = ' '.join(content.split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        
    def detect_changes(self, old_content: str, new_content: str) -> Tuple[bool, str]:
        """Detect and classify changes between EULA versions."""
        if not old_content or not new_content:
            return False, "Unable to compare content"
            
        # Calculate similarity ratio
        similarity = difflib.SequenceMatcher(None, old_content, new_content).ratio()
        
        if similarity > 0.95:
            return False, "No significant changes detected"
            
        # Generate diff for analysis
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            lineterm='',
            n=3
        )
        
        diff_text = ''.join(diff)
        return True, diff_text
        
    async def analyze_legal_impact(self, software: SoftwareAsset, diff_text: str) -> Dict:
        """Use LLM to analyze the legal impact of EULA changes."""
        
        analysis_prompt = PromptTemplate(
            input_variables=["software_name", "vendor", "diff_text"],
            template="""
            You are a legal technology expert analyzing changes to a software End User License Agreement (EULA).
            
            Software: {software_name}
            Vendor: {vendor}
            
            Changes detected:
            {diff_text}
            
            Please analyze these changes and provide:
            
            1. RISK_LEVEL: (LOW/MEDIUM/HIGH/CRITICAL) - Overall risk to the organization
            2. CHANGE_TYPE: Brief classification (e.g., "Licensing Terms", "Data Usage", "Termination Rights")
            3. IMPACT_SUMMARY: 2-3 sentence summary of key changes and their implications
            4. RECOMMENDED_ACTIONS: List of specific actions IT/Legal should take
            5. BUSINESS_IMPACT: How this might affect daily operations or compliance
            
            Focus on:
            - License scope changes
            - Data collection/usage modifications
            - Termination and cancellation terms
            - Liability and indemnification changes
            - Pricing or payment term modifications
            - Geographic or usage restrictions
            
            Format your response as JSON with the above fields.
            """
        )
        
        try:
            prompt_text = analysis_prompt.format(
                software_name=software.name,
                vendor=software.vendor,
                diff_text=diff_text[:8000]  # Limit diff text to avoid token limits
            )
            
            response = await self.llm.ainvoke(prompt_text)
            
            # Parse the JSON response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                logger.warning("Could not parse LLM response as JSON")
                return self._create_fallback_analysis(diff_text)
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._create_fallback_analysis(diff_text)
            
    def _create_fallback_analysis(self, diff_text: str) -> Dict:
        """Create a fallback analysis when LLM fails."""
        return {
            "RISK_LEVEL": "MEDIUM",
            "CHANGE_TYPE": "Unknown Changes",
            "IMPACT_SUMMARY": "EULA changes detected but could not be automatically analyzed. Manual review required.",
            "RECOMMENDED_ACTIONS": [
                "Perform manual legal review of changes",
                "Assess impact on current usage",
                "Consider legal consultation if business-critical"
            ],
            "BUSINESS_IMPACT": "Unknown - requires manual assessment"
        }
        
    async def check_software_asset(self, software: SoftwareAsset) -> Optional[EULAChange]:
        """Check a single software asset for EULA changes."""
        logger.info(f"Checking EULA for {software.name}")
        
        # Fetch current EULA content
        current_content = await self.fetch_eula_content(software.eula_url)
        if not current_content:
            logger.warning(f"Could not fetch EULA for {software.name}")
            return None
            
        # Calculate content hash
        current_hash = self.calculate_content_hash(current_content)
        
        # Check if content has changed
        if current_hash == software.current_hash:
            logger.info(f"No changes detected for {software.name}")
            software.last_checked = datetime.now()
            return None
            
        logger.info(f"EULA changes detected for {software.name}")
        
        # Get previous content (in real implementation, this would come from database)
        old_content = getattr(software, '_last_content', '')
        
        # Detect and analyze changes
        has_changes, diff_text = self.detect_changes(old_content, current_content)
        
        if not has_changes:
            # Update hash but no significant changes
            software.current_hash = current_hash
            software.last_checked = datetime.now()
            return None
            
        # Perform legal impact analysis
        analysis = await self.analyze_legal_impact(software, diff_text)
        
        # Create change record
        change = EULAChange(
            software=software,
            old_content=old_content,
            new_content=current_content,
            detected_at=datetime.now(),
            change_type=analysis.get("CHANGE_TYPE", "Unknown"),
            risk_level=RiskLevel(analysis.get("RISK_LEVEL", "medium").lower()),
            impact_summary=analysis.get("IMPACT_SUMMARY", "Changes detected"),
            recommended_actions=analysis.get("RECOMMENDED_ACTIONS", [])
        )
        
        # Update software asset
        software.current_hash = current_hash
        software.last_checked = datetime.now()
        software._last_content = current_content  # Store for next comparison
        
        return change
        
    async def monitor_all_assets(self):
        """Monitor all registered software assets."""
        logger.info(f"Starting monitoring cycle for {len(self.software_assets)} assets")
        
        changes = []
        for asset in self.software_assets.values():
            try:
                change = await self.check_software_asset(asset)
                if change:
                    changes.append(change)
                    await self.notify_stakeholders(change)
            except Exception as e:
                logger.error(f"Error monitoring {asset.name}: {e}")
                
        logger.info(f"Monitoring cycle completed. {len(changes)} changes detected.")
        return changes
        
    async def notify_stakeholders(self, change: EULAChange):
        """Send notifications to relevant stakeholders."""
        for handler in self.notification_handlers:
            try:
                await handler.send_notification(change)
            except Exception as e:
                logger.error(f"Notification failed: {e}")

class EmailNotificationHandler:
    """Email notification handler for EULA changes."""
    
    def __init__(self, smtp_config: Dict[str, str], recipients: List[str]):
        self.smtp_config = smtp_config
        self.recipients = recipients
        
    async def send_notification(self, change: EULAChange):
        """Send email notification about EULA change."""
        subject = f"EULA Change Alert: {change.software.name} - {change.risk_level.value.upper()}"
        
        body = f"""
        EULA Change Detected
        
        Software: {change.software.name}
        Vendor: {change.software.vendor}
        Risk Level: {change.risk_level.value.upper()}
        Change Type: {change.change_type}
        Detected: {change.detected_at.strftime('%Y-%m-%d %H:%M:%S')}
        
        Impact Summary:
        {change.impact_summary}
        
        Recommended Actions:
        """
        
        for i, action in enumerate(change.recommended_actions, 1):
            body += f"\n{i}. {action}"
            
        body += f"""
        
        EULA URL: {change.software.eula_url}
        
        Please review these changes and take appropriate action.
        
        ---
        Automated EULA Monitoring System
        """
        
        # In real implementation, use proper email library
        logger.info(f"Email notification sent for {change.software.name}")
        print(f"EMAIL NOTIFICATION:\nTo: {', '.join(self.recipients)}\nSubject: {subject}\n\n{body}")

class SlackNotificationHandler:
    """Slack notification handler for EULA changes."""
    
    def __init__(self, webhook_url: str, channel: str):
        self.webhook_url = webhook_url
        self.channel = channel
        
    async def send_notification(self, change: EULAChange):
        """Send Slack notification about EULA change."""
        
        color_map = {
            RiskLevel.LOW: "good",
            RiskLevel.MEDIUM: "warning", 
            RiskLevel.HIGH: "danger",
            RiskLevel.CRITICAL: "danger"
        }
        
        message = {
            "channel": self.channel,
            "attachments": [
                {
                    "color": color_map.get(change.risk_level, "warning"),
                    "title": f"EULA Change: {change.software.name}",
                    "fields": [
                        {
                            "title": "Risk Level",
                            "value": change.risk_level.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Change Type", 
                            "value": change.change_type,
                            "short": True
                        },
                        {
                            "title": "Impact Summary",
                            "value": change.impact_summary,
                            "short": False
                        }
                    ],
                    "footer": "EULA Monitor",
                    "ts": int(change.detected_at.timestamp())
                }
            ]
        }
        
        # In real implementation, send to Slack webhook
        logger.info(f"Slack notification sent for {change.software.name}")
        print(f"SLACK NOTIFICATION: {json.dumps(message, indent=2)}")

# Example usage and demonstration
async def main():
    """Demonstrate the EULA monitoring system."""
    
    # Initialize monitor (requires OpenAI API key)
    monitor = EULAMonitor(openai_api_key="your-openai-key-here")
    
    # Add notification handlers
    email_handler = EmailNotificationHandler(
        smtp_config={},
        recipients=["legal@company.com", "it-security@company.com"]
    )
    slack_handler = SlackNotificationHandler(
        webhook_url="https://hooks.slack.com/...",
        channel="#eula-alerts"
    )
    
    monitor.add_notification_handler(email_handler)
    monitor.add_notification_handler(slack_handler)
    
    # Register software assets
    sample_assets = [
        SoftwareAsset(
            name="Anaconda",
            vendor="Anaconda Inc.",
            version="2023.03",
            eula_url="https://www.anaconda.com/terms-of-service",
            last_checked=datetime.now() - timedelta(days=1),
            current_hash="",
            license_type="Commercial",
            business_critical=True
        ),
        SoftwareAsset(
            name="Docker Desktop",
            vendor="Docker Inc.",
            version="4.15.0",
            eula_url="https://www.docker.com/legal/docker-subscription-service-agreement/",
            last_checked=datetime.now() - timedelta(days=1), 
            current_hash="",
            license_type="Subscription",
            business_critical=True
        )
    ]
    
    for asset in sample_assets:
        monitor.register_software(asset)
    
    # Run monitoring cycle
    print("Starting EULA monitoring demonstration...")
    changes = await monitor.monitor_all_assets()
    
    if changes:
        print(f"\nDetected {len(changes)} EULA changes:")
        for change in changes:
            print(f"- {change.software.name}: {change.change_type} ({change.risk_level.value})")
    else:
        print("\nNo EULA changes detected in this cycle.")

def setup_scheduled_monitoring(monitor: EULAMonitor, frequency_hours: int = 24):
    """Setup scheduled monitoring using the schedule library."""
    
    def run_monitoring():
        """Wrapper to run async monitoring in sync context."""
        asyncio.run(monitor.monitor_all_assets())
    
    # Schedule monitoring
    schedule.every(frequency_hours).hours.do(run_monitoring)
    
    print(f"Scheduled monitoring every {frequency_hours} hours")
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    # Run the demonstration
    print("EULA Monitoring System - Technical Implementation Example")
    print("=" * 60)
    
    # Note: This is a demonstration - replace with actual OpenAI API key
    print("This is a demonstration of the core EULA monitoring functionality.")
    print("To run with real monitoring, provide a valid OpenAI API key.")
    
    # Uncomment to run the actual demonstration
    # asyncio.run(main())
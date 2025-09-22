#!/usr/bin/env python3
"""
EULA Monitoring System - Standalone Demo
This script demonstrates the core concepts of the EULA monitoring system
without requiring external API keys or dependencies.
"""

import hashlib
import json
import difflib
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MockSoftwareAsset:
    """Mock software asset for demonstration."""
    name: str
    vendor: str
    version: str
    eula_url: str
    last_checked: datetime
    current_hash: str
    license_type: str
    business_critical: bool = False

@dataclass  
class MockEULAChange:
    """Mock EULA change for demonstration."""
    software_name: str
    old_content_sample: str
    new_content_sample: str
    detected_at: datetime
    change_type: str
    risk_level: RiskLevel
    impact_summary: str
    recommended_actions: List[str]

class MockEULAAnalyzer:
    """Mock EULA analyzer that simulates the real system behavior."""
    
    def __init__(self):
        self.mock_eulas = {
            "Anaconda": {
                "old": """
                ANACONDA INDIVIDUAL EDITION END USER LICENSE AGREEMENT
                
                Anaconda Individual Edition is free to use for personal, educational, 
                and non-commercial purposes. Commercial use requires a paid license.
                
                You may use Anaconda Individual Edition on unlimited computers for:
                - Personal projects
                - Educational coursework
                - Academic research
                - Non-profit work
                
                Commercial use is defined as any use that generates revenue or 
                supports business operations.
                """,
                "new": """
                ANACONDA INDIVIDUAL EDITION END USER LICENSE AGREEMENT
                
                Anaconda Individual Edition is free to use for personal, educational, 
                and limited non-commercial purposes. Commercial use requires a paid license.
                
                You may use Anaconda Individual Edition on up to 5 computers for:
                - Personal projects  
                - Educational coursework
                - Academic research (with restrictions)
                - Non-profit work (limited scope)
                
                Commercial use is defined as any use that generates revenue, 
                supports business operations, or involves more than 200 employees.
                """
            },
            "Docker Desktop": {
                "old": """
                DOCKER DESKTOP LICENSE AGREEMENT
                
                Docker Desktop is free for personal use, education, and small businesses.
                Organizations with more than 250 employees or $10M in revenue require 
                a paid subscription.
                
                Free usage includes:
                - Personal development
                - Educational purposes
                - Small business use
                - Open source projects
                """,
                "new": """
                DOCKER DESKTOP LICENSE AGREEMENT
                
                Docker Desktop is free for personal use and education only.
                All business use requires a paid subscription.
                
                Free usage includes:
                - Personal development only
                - Educational purposes (academic institutions)
                
                Business use is defined as any use in a commercial environment,
                regardless of company size or revenue.
                """
            }
        }
    
    def calculate_hash(self, content: str) -> str:
        """Calculate content hash."""
        normalized = ' '.join(content.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def detect_changes(self, software_name: str) -> Optional[MockEULAChange]:
        """Simulate change detection."""
        if software_name not in self.mock_eulas:
            return None
            
        old_content = self.mock_eulas[software_name]["old"]
        new_content = self.mock_eulas[software_name]["new"]
        
        old_hash = self.calculate_hash(old_content)
        new_hash = self.calculate_hash(new_content)
        
        if old_hash == new_hash:
            return None
            
        # Generate diff
        diff = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile='old_eula.txt',
            tofile='new_eula.txt',
            n=3
        ))
        
        # Analyze changes based on software
        analysis = self.mock_analyze_impact(software_name, old_content, new_content)
        
        return MockEULAChange(
            software_name=software_name,
            old_content_sample=old_content[:200] + "...",
            new_content_sample=new_content[:200] + "...",
            detected_at=datetime.now(),
            change_type=analysis["change_type"],
            risk_level=RiskLevel(analysis["risk_level"]),
            impact_summary=analysis["impact_summary"],
            recommended_actions=analysis["recommended_actions"]
        )
    
    def mock_analyze_impact(self, software_name: str, old_content: str, new_content: str) -> Dict:
        """Mock LLM analysis of EULA changes."""
        
        # Simple rule-based analysis for demonstration
        if software_name == "Anaconda":
            return {
                "change_type": "License Scope Restrictions",
                "risk_level": "high",
                "impact_summary": "Significant restrictions added to Individual Edition license. Computer limit introduced (5 max), academic use restricted, and commercial definition expanded to include companies with 200+ employees.",
                "recommended_actions": [
                    "Audit current Anaconda usage across the organization",
                    "Count active installations and users",
                    "Evaluate need for commercial license purchase", 
                    "Consider migration to alternative tools if cost-prohibitive",
                    "Update software policies to reflect new restrictions"
                ]
            }
        elif software_name == "Docker Desktop":
            return {
                "change_type": "Commercial Use Policy",
                "risk_level": "critical", 
                "impact_summary": "Docker Desktop now requires paid subscription for ALL business use, regardless of company size. This eliminates the previous exemption for smaller organizations.",
                "recommended_actions": [
                    "IMMEDIATE: Assess all Docker Desktop installations",
                    "Calculate licensing costs for current usage",
                    "Evaluate Docker alternatives (Podman, Rancher Desktop)",
                    "Begin migration planning if staying with Docker",
                    "Update procurement and software approval processes"
                ]
            }
        else:
            return {
                "change_type": "General Changes",
                "risk_level": "medium",
                "impact_summary": "Changes detected in EULA terms requiring review.",
                "recommended_actions": [
                    "Review changes with legal team",
                    "Assess impact on current usage",
                    "Update compliance documentation"
                ]
            }

def format_change_report(change: MockEULAChange) -> str:
    """Format a change report for display."""
    risk_emoji = {
        RiskLevel.LOW: "🟢",
        RiskLevel.MEDIUM: "🟡", 
        RiskLevel.HIGH: "🟠",
        RiskLevel.CRITICAL: "🔴"
    }
    
    report = f"""
{'='*60}
{risk_emoji[change.risk_level]} EULA CHANGE ALERT - {change.risk_level.value.upper()}
{'='*60}

Software: {change.software_name}
Change Type: {change.change_type}
Detected: {change.detected_at.strftime('%Y-%m-%d %H:%M:%S')}

IMPACT SUMMARY:
{change.impact_summary}

RECOMMENDED ACTIONS:
"""
    
    for i, action in enumerate(change.recommended_actions, 1):
        report += f"{i}. {action}\n"
    
    report += "\n" + "="*60
    return report

def demo_notification_system(change: MockEULAChange):
    """Demonstrate the notification system."""
    print("\n📧 EMAIL NOTIFICATION SIMULATION:")
    print("─" * 50)
    print(f"To: legal-team@company.com, it-security@company.com")
    print(f"Subject: URGENT: EULA Change - {change.software_name} ({change.risk_level.value.upper()})")
    print(f"Priority: {'High' if change.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else 'Normal'}")
    
    print("\n💬 SLACK NOTIFICATION SIMULATION:")
    print("─" * 50)
    slack_msg = {
        "channel": "#eula-alerts",
        "text": f"🚨 EULA Change Detected: {change.software_name}",
        "attachments": [{
            "color": "danger" if change.risk_level == RiskLevel.CRITICAL else "warning",
            "fields": [
                {"title": "Risk Level", "value": change.risk_level.value.upper(), "short": True},
                {"title": "Change Type", "value": change.change_type, "short": True},
                {"title": "Impact", "value": change.impact_summary[:100] + "...", "short": False}
            ]
        }]
    }
    print(json.dumps(slack_msg, indent=2))

def main():
    """Run the EULA monitoring demonstration."""
    print("🔍 REAL-TIME SOFTWARE EULA ANALYSIS - DEMONSTRATION")
    print("=" * 60)
    print("This demo simulates the EULA monitoring system detecting")
    print("changes in software licenses and providing analysis.")
    print()
    
    # Initialize mock analyzer
    analyzer = MockEULAAnalyzer()
    
    # Mock software assets
    assets = [
        MockSoftwareAsset(
            name="Anaconda",
            vendor="Anaconda Inc.",
            version="2023.03",
            eula_url="https://www.anaconda.com/terms-of-service",
            last_checked=datetime.now() - timedelta(hours=24),
            current_hash="abc123",
            license_type="Individual",
            business_critical=True
        ),
        MockSoftwareAsset(
            name="Docker Desktop", 
            vendor="Docker Inc.",
            version="4.15.0",
            eula_url="https://www.docker.com/legal/docker-subscription-service-agreement/",
            last_checked=datetime.now() - timedelta(hours=12),
            current_hash="def456",
            license_type="Free/Commercial",
            business_critical=True
        )
    ]
    
    print("📋 REGISTERED SOFTWARE ASSETS:")
    print("─" * 50)
    for asset in assets:
        critical_marker = "⚠️ CRITICAL" if asset.business_critical else "ℹ️ Standard"
        print(f"• {asset.name} ({asset.vendor}) - {critical_marker}")
    
    print(f"\n🔄 STARTING MONITORING CYCLE...")
    print("─" * 50)
    
    detected_changes = []
    
    # Check each asset for changes
    for asset in assets:
        print(f"Checking {asset.name}...")
        change = analyzer.detect_changes(asset.name)
        
        if change:
            detected_changes.append(change)
            print(f"  ✓ Changes detected - Risk Level: {change.risk_level.value.upper()}")
        else:
            print(f"  ✓ No changes detected")
    
    # Report findings
    if detected_changes:
        print(f"\n🚨 MONITORING RESULTS: {len(detected_changes)} CHANGES DETECTED")
        
        for change in detected_changes:
            print(format_change_report(change))
            demo_notification_system(change)
            print("\n" + "─" * 60 + "\n")
            
    else:
        print(f"\n✅ MONITORING COMPLETE: No EULA changes detected")
    
    # Summary and next steps
    print("📊 MONITORING SUMMARY:")
    print("─" * 50)
    print(f"Assets Monitored: {len(assets)}")
    print(f"Changes Detected: {len(detected_changes)}")
    critical_changes = sum(1 for c in detected_changes if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
    print(f"Critical Changes: {critical_changes}")
    print(f"Next Monitoring Cycle: {(datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')}")
    
    if detected_changes:
        print(f"\n⚠️  IMMEDIATE ACTIONS REQUIRED:")
        print("─" * 50)
        for change in detected_changes:
            if change.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                print(f"• {change.software_name}: {change.recommended_actions[0]}")
    
    print(f"\n✅ Demo completed successfully!")
    print("For production deployment, configure API keys and notification channels.")

if __name__ == "__main__":
    main()
# DL_AI Repository

## Projects

### 1. Database Chat Application (RAG Demo)
Repository for Demo for talking to Database using natural language processing.

Location: `RAG_Demo_Streamlit/`

### 2. Real-time Software EULA Analysis Solution

A comprehensive agentic solution for monitoring End User License Agreements (EULAs) of software in use at organizations. This system provides automated monitoring, legal text analysis, and proactive notifications to ensure compliance and minimize risk exposure.

#### Key Features
- **Real-time Monitoring**: Continuous tracking of EULA changes for registered software
- **AI-Powered Analysis**: Legal language interpretation using advanced LLMs
- **Risk Assessment**: Automated risk scoring and impact analysis
- **Proactive Notifications**: Multi-channel alerts (email, Slack, Teams)
- **Compliance Tracking**: Audit trail and compliance reporting
- **Configurable Monitoring**: User-defined monitoring frequencies

#### Files
- `EULA_Analysis_Solution_Document.md` - Comprehensive technical specification
- `eula_monitoring_implementation.py` - Core implementation example
- `eula_monitor_config.toml` - Configuration file template

#### Use Case
Addresses the challenge of monitoring EULA changes for software like Anaconda, Docker, and other enterprise tools to ensure licensing compliance and prevent costly violations.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the configuration template and customize for your environment
2. Set up environment variables as specified in the config file
3. Configure notification channels and monitoring frequencies
4. Register software assets for monitoring

## Usage

### Database Chat Application
```bash
cd RAG_Demo_Streamlit/
streamlit run NLP_Chat_to_Database.py
```

### EULA Monitoring System
```bash
python eula_monitoring_implementation.py
```

## Architecture

The EULA Analysis solution uses:
- **LLMs**: GPT-4 Turbo, Claude-3 for legal text analysis
- **Web Scraping**: Playwright/Selenium for content fetching
- **Databases**: PostgreSQL for data storage, Redis for caching
- **Notifications**: Email, Slack, Teams integration
- **Monitoring**: Configurable scheduling and alerting

## Security

- Encryption at rest and in transit
- API key management
- Audit logging
- GDPR compliance features

## License

This project contains examples and demonstrations for educational and enterprise use.


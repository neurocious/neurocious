# BlessedRSI.Web - Faith-Based Investment Platform

A production-grade Blazor Server application that combines biblical wisdom with cutting-edge Epistemic RSI technology for investment education and community fellowship.

## Architecture Overview

- **Frontend**: Blazor Server with Bootstrap 5
- **Backend**: ASP.NET Core 8 with Entity Framework Core
- **Database**: SQL Server
- **Real-time**: SignalR for community features
- **External API**: DigitalOcean microservice for E-RSI calculations
- **Payments**: Stripe integration for subscriptions

## Project Structure

```
BlessedRSI.Web/
├── Components/
│   ├── Layout/
│   │   └── MainLayout.razor          # Main application layout
│   └── Pages/
│       ├── Home.razor               # Landing page
│       ├── Backtest.razor           # Strategy testing interface
│       ├── Community.razor          # Community discussion
│       └── Leaderboard.razor        # Performance rankings
├── Data/
│   └── ApplicationDbContext.cs      # Entity Framework context
├── Models/
│   ├── ApplicationUser.cs           # Extended user model
│   ├── BacktestModels.cs           # Trading and backtest models
│   └── CommunityModels.cs          # Community and social features
├── Services/
│   ├── TradingApiService.cs        # DigitalOcean API integration
│   ├── BacktestService.cs          # Backtesting business logic
│   ├── CommunityService.cs         # Community features
│   ├── LeaderboardService.cs       # Rankings and statistics
│   └── SubscriptionService.cs      # Stripe payment processing
├── Hubs/
│   └── CommunityHub.cs             # SignalR real-time features
└── wwwroot/
    └── css/
        └── site.css                # Custom styling
```

## Features

### Core Platform Features
- **E-RSI Strategy Testing**: Interactive parameter adjustment with real-time backtesting
- **Community Fellowship**: Discussion forums with biblical integration
- **Leaderboard System**: Performance rankings with achievement system
- **Subscription Tiers**: Free to $299/month with progressive features

### Subscription Tiers
1. **Seeker (Free)**: Basic education, 3 backtests/day, community access
2. **Believer ($29/month)**: Unlimited backtests, advanced analytics
3. **Disciple ($99/month)**: Real-time E-RSI, custom alerts
4. **Apostle ($299/month)**: API access, white-label solutions

### Biblical Integration
- Daily devotionals tied to market principles
- Scripture-based strategy naming
- Community prayer requests and testimonials
- Achievement system with biblical themes

## Setup Instructions

### Prerequisites
- .NET 8 SDK
- SQL Server (LocalDB for development)
- Visual Studio 2022 or VS Code

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BlessedRSI.Web
   ```

2. **Configure appsettings**
   Update `appsettings.Development.json`:
   ```json
   {
     "ConnectionStrings": {
       "DefaultConnection": "Server=(localdb)\\mssqllocaldb;Database=BlessedRSIDb_Dev;Trusted_Connection=true;MultipleActiveResultSets=true"
     },
     "TradingAPI": {
       "BaseUrl": "http://localhost:8000",
       "ApiKey": "your-api-key"
     },
     "Stripe": {
       "PublishableKey": "pk_test_...",
       "SecretKey": "sk_test_...",
       "WebhookSecret": "whsec_..."
     }
   }
   ```

3. **Create database migrations**
   ```bash
   dotnet ef migrations add InitialCreate
   dotnet ef database update
   ```

4. **Install dependencies**
   ```bash
   dotnet restore
   ```

5. **Run the application**
   ```bash
   dotnet run
   ```

## DigitalOcean API Integration

The platform integrates with a Python FastAPI microservice for E-RSI calculations:

### Expected API Endpoints
- `POST /api/backtest` - Run strategy backtest
- `GET /api/symbols` - Get available trading symbols
- `GET /api/ersi/{symbol}` - Get current Epistemic RSI value

### API Request/Response Format
```json
// Backtest Request
{
  "buyThreshold": 40.0,
  "sellTrigger": 90.0,
  "sellDelayWeeks": 4,
  "positionSizePct": 0.24,
  "stopLossPct": 0.24,
  "startingCapital": 100000
}

// Backtest Response
{
  "success": true,
  "data": {
    "totalReturn": 1.7986,
    "sortinoRatio": 8.644,
    "sharpeRatio": 4.23,
    "winRate": 0.708,
    "totalTrades": 125,
    "winningTrades": 88
  }
}
```

## Database Schema

### Key Entities
- **ApplicationUser**: Extended Identity user with subscription info
- **BacktestResult**: Strategy test results and parameters
- **CommunityPost**: Discussion posts with biblical integration
- **Achievement**: Gamification system with biblical themes
- **StrategyShare**: Community-shared strategies

### Sample Data Seeding
The application includes seed data for:
- Achievement definitions ("Blessed Beginner", "David's Courage", etc.)
- Sample community posts
- Default system users

## Deployment

### Production Configuration
1. Update connection strings for production SQL Server
2. Configure Stripe production keys
3. Set up SSL certificates
4. Configure logging and monitoring

### Environment Variables
```bash
ASPNETCORE_ENVIRONMENT=Production
ConnectionStrings__DefaultConnection="Server=..."
TradingAPI__BaseUrl="https://api.blessedrsi.com"
TradingAPI__ApiKey="production-key"
Stripe__SecretKey="sk_live_..."
```

## Development Notes

### Architecture Decisions
- **Blazor Server**: Chosen for real-time capabilities and reduced client complexity
- **Entity Framework**: Code-first approach with comprehensive relationships
- **SignalR**: Enables real-time community features and notifications
- **Bootstrap 5**: Modern, accessible UI framework

### Performance Considerations
- Implement caching for leaderboard data
- Optimize database queries with proper indexing
- Use connection pooling for external API calls
- Implement rate limiting for free tier users

### Security Features
- ASP.NET Core Identity for authentication
- Role-based authorization
- API key security for external services
- Input validation and sanitization

## Contributing

1. Follow established naming conventions
2. Add biblical themes where appropriate
3. Maintain responsive design principles
4. Include comprehensive error handling
5. Write tests for critical business logic

## License

This project is proprietary software for BlessedRSI.com.

---

*"Trust in the Lord with all your heart and lean not on your own understanding."* - Proverbs 3:5
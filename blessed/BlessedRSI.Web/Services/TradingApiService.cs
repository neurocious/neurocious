using System.Text.Json;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class TradingApiService
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<TradingApiService> _logger;

    public TradingApiService(IHttpClientFactory httpClientFactory, ILogger<TradingApiService> logger)
    {
        _httpClient = httpClientFactory.CreateClient("TradingAPI");
        _logger = logger;
    }

    public async Task<BacktestResult?> RunBacktestAsync(BacktestParameters parameters)
    {
        try
        {
            var response = await _httpClient.PostAsJsonAsync("/api/backtest", parameters);
            response.EnsureSuccessStatusCode();
            
            var result = await response.Content.ReadFromJsonAsync<BacktestApiResponse>();
            
            if (result?.Success == true && result.Data != null)
            {
                return new BacktestResult
                {
                    TotalReturn = result.Data.TotalReturn,
                    SortinoRatio = result.Data.SortinoRatio,
                    SharpeRatio = result.Data.SharpeRatio,
                    WinRate = result.Data.WinRate,
                    MaxDrawdown = result.Data.MaxDrawdown,
                    AnnualizedReturn = result.Data.AnnualizedReturn,
                    DownsideVolatility = result.Data.DownsideVolatility,
                    TotalTrades = result.Data.TotalTrades,
                    WinningTrades = result.Data.WinningTrades,
                    BuyThreshold = parameters.BuyThreshold,
                    SellTrigger = parameters.SellTrigger,
                    SellDelayWeeks = parameters.SellDelayWeeks,
                    PositionSizePct = parameters.PositionSizePct,
                    StopLossPct = parameters.StopLossPct,
                    StartingCapital = parameters.StartingCapital,
                    DetailedResultsJson = JsonSerializer.Serialize(result.Data.Trades)
                };
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running backtest");
        }
        
        return null;
    }

    public async Task<List<string>> GetAvailableSymbolsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/api/symbols");
            response.EnsureSuccessStatusCode();
            
            var result = await response.Content.ReadFromJsonAsync<List<string>>();
            return result ?? new List<string>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching symbols");
            return new List<string>();
        }
    }

    public async Task<decimal> GetCurrentEpistemicRsiAsync(string symbol)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/ersi/{symbol}");
            response.EnsureSuccessStatusCode();
            
            var result = await response.Content.ReadFromJsonAsync<EpistemicRsiResponse>();
            return result?.Value ?? 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching E-RSI for {Symbol}", symbol);
            return 0;
        }
    }
}

public class BacktestApiResponse
{
    public bool Success { get; set; }
    public string? Message { get; set; }
    public BacktestApiData? Data { get; set; }
}

public class BacktestApiData
{
    public decimal TotalReturn { get; set; }
    public decimal SortinoRatio { get; set; }
    public decimal SharpeRatio { get; set; }
    public decimal WinRate { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal AnnualizedReturn { get; set; }
    public decimal DownsideVolatility { get; set; }
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public List<TradeResult> Trades { get; set; } = new();
}

public class TradeResult
{
    public string Symbol { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public DateTime Date { get; set; }
    public decimal EpistemicRsi { get; set; }
    public decimal ProfitLoss { get; set; }
    public decimal ProfitLossPercent { get; set; }
}

public class EpistemicRsiResponse
{
    public decimal Value { get; set; }
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
}
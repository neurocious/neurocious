using Microsoft.EntityFrameworkCore;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;
using Stripe;
using Stripe.Checkout;

namespace BlessedRSI.Web.Services;

public class SubscriptionService
{
    private readonly ApplicationDbContext _context;
    private readonly IConfiguration _configuration;

    public SubscriptionService(ApplicationDbContext context, IConfiguration configuration)
    {
        _context = context;
        _configuration = configuration;
        StripeConfiguration.ApiKey = _configuration["Stripe:SecretKey"];
    }

    public async Task<string> CreateCheckoutSessionAsync(string userId, SubscriptionTier tier)
    {
        var user = await _context.Users.FindAsync(userId);
        if (user == null) throw new ArgumentException("User not found");

        var priceId = GetStripePriceId(tier);
        var options = new SessionCreateOptions
        {
            PaymentMethodTypes = new List<string> { "card" },
            LineItems = new List<SessionLineItemOptions>
            {
                new SessionLineItemOptions
                {
                    Price = priceId,
                    Quantity = 1,
                }
            },
            Mode = "subscription",
            SuccessUrl = $"{_configuration["Domain"]}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}",
            CancelUrl = $"{_configuration["Domain"]}/subscription/cancel",
            ClientReferenceId = userId,
            CustomerEmail = user.Email,
            Metadata = new Dictionary<string, string>
            {
                { "user_id", userId },
                { "subscription_tier", tier.ToString() }
            }
        };

        var service = new SessionService();
        var session = await service.CreateAsync(options);
        
        return session.Url;
    }

    public async Task<bool> ProcessWebhookAsync(string json, string signature)
    {
        try
        {
            var webhookSecret = _configuration["Stripe:WebhookSecret"];
            var stripeEvent = EventUtility.ConstructEvent(json, signature, webhookSecret);

            switch (stripeEvent.Type)
            {
                case Events.CheckoutSessionCompleted:
                    await HandleCheckoutSessionCompletedAsync(stripeEvent);
                    break;
                case Events.InvoicePaymentSucceeded:
                    await HandleInvoicePaymentSucceededAsync(stripeEvent);
                    break;
                case Events.CustomerSubscriptionDeleted:
                    await HandleSubscriptionDeletedAsync(stripeEvent);
                    break;
            }

            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    private async Task HandleCheckoutSessionCompletedAsync(Event stripeEvent)
    {
        var session = stripeEvent.Data.Object as Session;
        if (session?.ClientReferenceId == null) return;

        var userId = session.ClientReferenceId;
        var user = await _context.Users.FindAsync(userId);
        if (user == null) return;

        if (session.Metadata.TryGetValue("subscription_tier", out var tierString) &&
            Enum.TryParse<SubscriptionTier>(tierString, out var tier))
        {
            user.SubscriptionTier = tier;
            user.SubscriptionExpiresAt = DateTime.UtcNow.AddMonths(1);
            await _context.SaveChangesAsync();
        }
    }

    private async Task HandleInvoicePaymentSucceededAsync(Event stripeEvent)
    {
        var invoice = stripeEvent.Data.Object as Invoice;
        if (invoice?.CustomerId == null) return;

        // Extend subscription for existing customers
        var user = await _context.Users
            .FirstOrDefaultAsync(u => u.Email == invoice.CustomerEmail);
        
        if (user != null && user.SubscriptionTier != SubscriptionTier.Sparrow)
        {
            user.SubscriptionExpiresAt = DateTime.UtcNow.AddMonths(1);
            await _context.SaveChangesAsync();
        }
    }

    private async Task HandleSubscriptionDeletedAsync(Event stripeEvent)
    {
        var subscription = stripeEvent.Data.Object as Subscription;
        if (subscription?.CustomerId == null) return;

        var customerService = new CustomerService();
        var customer = await customerService.GetAsync(subscription.CustomerId);
        
        var user = await _context.Users
            .FirstOrDefaultAsync(u => u.Email == customer.Email);
        
        if (user != null)
        {
            user.SubscriptionTier = SubscriptionTier.Sparrow;
            user.SubscriptionExpiresAt = null;
            await _context.SaveChangesAsync();
        }
    }

    private string GetStripePriceId(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Lion => _configuration["Stripe:LionPriceId"] ?? "price_lion",
            SubscriptionTier.Eagle => _configuration["Stripe:EaglePriceId"] ?? "price_eagle",
            SubscriptionTier.Shepherd => _configuration["Stripe:ShepherdPriceId"] ?? "price_shepherd",
            _ => throw new ArgumentException("Invalid subscription tier")
        };
    }

    public async Task<bool> IsSubscriptionActiveAsync(string userId)
    {
        var user = await _context.Users.FindAsync(userId);
        if (user == null) return false;

        return user.SubscriptionTier != SubscriptionTier.Sparrow &&
               (user.SubscriptionExpiresAt == null || user.SubscriptionExpiresAt > DateTime.UtcNow);
    }

    public async Task<SubscriptionTier> GetUserSubscriptionTierAsync(string userId)
    {
        var user = await _context.Users.FindAsync(userId);
        if (user == null) return SubscriptionTier.Sparrow;

        // Check if subscription is expired
        if (user.SubscriptionExpiresAt != null && user.SubscriptionExpiresAt <= DateTime.UtcNow)
        {
            user.SubscriptionTier = SubscriptionTier.Sparrow;
            user.SubscriptionExpiresAt = null;
            await _context.SaveChangesAsync();
        }

        return user.SubscriptionTier;
    }
}
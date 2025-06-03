using Microsoft.AspNetCore.SignalR;
using Microsoft.AspNetCore.Authorization;

namespace BlessedRSI.Web.Hubs;

[Authorize]
public class CommunityHub : Hub
{
    public async Task JoinCommunity()
    {
        await Groups.AddToGroupAsync(Context.ConnectionId, "Community");
        await Clients.Group("Community").SendAsync("UserJoined", Context.User?.Identity?.Name);
    }

    public async Task LeaveCommunity()
    {
        await Groups.RemoveFromGroupAsync(Context.ConnectionId, "Community");
        await Clients.Group("Community").SendAsync("UserLeft", Context.User?.Identity?.Name);
    }

    public async Task SendMessage(string message)
    {
        await Clients.Group("Community").SendAsync("ReceiveMessage", Context.User?.Identity?.Name, message);
    }

    public async Task NotifyNewPost(string postTitle, string userName)
    {
        await Clients.Group("Community").SendAsync("NewPost", postTitle, userName);
    }

    public async Task NotifyNewComment(int postId, string userName)
    {
        await Clients.Group("Community").SendAsync("NewComment", postId, userName);
    }

    public async Task NotifyAchievementEarned(string userName, string achievementName)
    {
        await Clients.Group("Community").SendAsync("AchievementEarned", userName, achievementName);
    }

    public override async Task OnConnectedAsync()
    {
        await JoinCommunity();
        await base.OnConnectedAsync();
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        await LeaveCommunity();
        await base.OnDisconnectedAsync(exception);
    }
}
import click
import json
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from mechanex import _mx as mx

console = Console()
CONFIG_DIR = Path.home() / ".mechanex"
CONFIG_FILE = CONFIG_DIR / "config.json"

def save_config(config):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

@click.group()
@click.pass_context
def main(ctx):
    """Mechanex CLI for managing your Axionic account and models."""
    config = load_config()
    if "api_key" in config:
        mx.set_key(config["api_key"])
    ctx.obj = config

@main.command()
@click.option('--email', prompt='Email', help='Your email address.')
@click.option('--password', prompt=True, hide_input=True, help='Your password.')
def signup(email, password):
    """Sign up for a new Axionic account."""
    try:
        mx.signup(email, password)
        console.print("[bold green]Successfully signed up![/bold green] You can now log in.")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@main.command()
@click.option('--email', prompt='Email', help='Your email address.')
@click.option('--password', prompt=True, hide_input=True, help='Your password.')
@click.pass_obj
def login(obj, email, password):
    """Log in to your Axionic account."""
    try:
        result = mx.login(email, password)
        token = result.get("session", {}).get("access_token")
        if token:
            obj["api_key"] = token
            save_config(obj)
            console.print(Panel("[bold green]Successfully logged in![/bold green]\nAPI key saved to ~/.mechanex/config.json", title="Welcome"))
        else:
            console.print("[yellow]Login successful, but no access token received.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@main.command()
def list_api_keys():
    """List your Axionic API keys."""
    try:
        keys = mx.list_api_keys()
        
        # Handle the case where the API might return an object with a 'keys' list
        keys_list = keys if isinstance(keys, list) else keys.get("keys", [])
        
        if not keys_list:
            console.print("[yellow]No API keys found.[/yellow]")
            return

        table = Table(title="Axionic API Keys", header_style="bold magenta")
        table.add_column("Name", style="bold white")
        table.add_column("ID", style="dim")
        table.add_column("Key", style="cyan", overflow="fold")
        table.add_column("Created At", style="green")
        table.add_column("Last Used", style="blue")

        for k in keys_list:
            display_name = k.get("name") or "(none)"
            key_val = k.get("key") or k.get("api_key", "N/A")
            
            created_at = k.get("created_at", "N/A")
            if created_at and "T" in created_at:
                created_at = created_at.split("T")[0]
                
            last_used = k.get("last_used") or "Never"
            if last_used and "T" in last_used:
                last_used = last_used.split("T")[0]

            table.add_row(
                display_name,
                k.get("id", "N/A"),
                key_val,
                created_at,
                last_used
            )
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@main.command()
@click.option('--name', default='Default Key', help='Name for the new API key.')
def create_api_key(name):
    """Create a new Axionic API key."""
    try:
        result = mx.create_api_key(name)
        console.print(f"[bold green]Successfully created API key:[/bold green] {name}")
        
        key_val = result.get("key", result.get("api_key", "Check response"))
        console.print(Panel(f"Key: [bold cyan]{key_val}[/bold cyan]\n\n[dim italic]Store this securely; it will not be shown again.[/dim italic]", title="New API Key", border_style="green"))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@main.command()
def whoami():
    """Show the current logged-in user and profile info."""
    config = load_config()
    if "api_key" in config:
        try:
            # Try new whoami endpoint
            user = mx.whoami()
            
            user_info = f"User: [bold cyan]{user.get('email', 'Unknown')}[/bold cyan]\n"
            user_info += f"ID: [dim]{user.get('id', 'N/A')}[/dim]\n"
            
            # Optionally get dashboard info for orgs/models
            try:
                dashboard = mx._get("/auth/dashboard")
                orgs = dashboard.get("organizations", [])
                models = dashboard.get("active_models", [])
                
                user_info += f"Organizations: [magenta]{len(orgs)}[/magenta]\n"
                user_info += f"Active Models: [blue]{len(models)}[/blue]"
                
                console.print(Panel(user_info, title="[bold]Current Session[/bold]", border_style="cyan"))
                
                if orgs:
                    org_table = Table(title="Organizations", box=None)
                    org_table.add_column("Name", style="bold")
                    org_table.add_column("Role", style="italic")
                    for org in orgs:
                        org_data = org.get("organization", org)
                        org_table.add_row(org_data.get("name", "N/A"), org.get("role", "member"))
                    console.print(org_table)
            except:
                # If dashboard fails, just show basic user info
                console.print(Panel(user_info, title="[bold]Current Session[/bold]", border_style="cyan"))
                
        except Exception as e:
            # Fallback for old backends or connectivity issues
            console.print(Panel(
                f"[green]Authenticated[/green]\n[dim]API Key is present but could not fetch profile: {str(e)}[/dim]",
                title="Session Status",
                border_style="yellow"
            ))
    else:
        console.print("[bold yellow]Not logged in.[/bold yellow] Run 'mechanex login' to begin.")

@main.command()
def logout():
    """Log out and remove stored credentials."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        console.print("[bold green]Logged out successfully.[/bold green]")
    else:
        console.print("You are not logged in.")

if __name__ == "__main__":
    main()

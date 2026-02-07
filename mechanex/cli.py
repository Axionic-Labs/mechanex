import click
import json
import os
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from mechanex import _mx as mx
import huggingface_hub

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
    
    # Load API Key if present
    if "api_key" in config:
        mx.set_key(config["api_key"])
        
    # Load Access Token (JWT) if present
    if "access_token" in config:
        refresh_token = config.get("refresh_token")
        mx.set_token(config["access_token"], refresh_token=refresh_token)
        
    ctx.obj = config

@main.command()
@click.option('--email', prompt='Email', help='Your email address.')
@click.option('--password', prompt=True, hide_input=True, help='Your password.')
@click.pass_obj
def signup(obj, email, password):
    """Sign up for a new Axionic account, log in, and generate an API key."""
    try:
        # 1. Signup
        signup_result = mx.signup(email, password)

        console.print("[bold green]Successfully signed up![/bold green]")
        
        if signup_result is None:
            signup_result = {}

        session = signup_result.get("session", {})
        
        access_token = None
        refresh_token = None

        if session and session.get("access_token"):
            access_token = session.get("access_token")
            refresh_token = session.get("refresh_token")
            console.print("Authenticated via signup response.")
        else:
            # 2. Auto-Login with retry
            console.print("Logging in to generate credentials...")
            time.sleep(1) 
            
            try:
                login_result = mx.login(email, password)
                session = login_result.get("session", {})
            except Exception:
                # Retry once
                console.print("Retrying login...")
                time.sleep(2)
                login_result = mx.login(email, password)
                session = login_result.get("session", {})

            access_token = session.get("access_token")
            refresh_token = session.get("refresh_token")
        
        if not access_token:
            console.print("[yellow]Signup successful, but could not log in automatically.[/yellow]")
            return
            
        # Set session tokens
        mx.set_token(access_token, refresh_token=refresh_token)
        obj["access_token"] = access_token
        if refresh_token:
            obj["refresh_token"] = refresh_token
        save_config(obj)

        # 3. Generate Token
        console.print("Creating default API key...")
        # Note: mx.login() or manual set_key() above sets mx.api_key to the session token
        key_result = mx.create_api_key(name="Default Key")
        new_key = key_result.get("key") or key_result.get("api_key")
        
        if new_key:
            # 4. Save Permanent Key
            obj["api_key"] = new_key
            save_config(obj)
            
            console.print(Panel(
                f"[bold green]Account Setup Complete![/bold green]\n\n"
                f"Generated API Key: [bold cyan]{new_key}[/bold cyan]\n",
                title="Welcome to Mechanex",
                border_style="green"
            ))

            # 5. Hugging Face Login
            if click.confirm("\nDo you want to log in to Hugging Face now?", default=True):
                hf_token = click.prompt("Hugging Face Token", hide_input=True)
                try:
                    huggingface_hub.login(token=hf_token)
                    console.print("[bold green]Successfully logged in to Hugging Face![/bold green]")
                except Exception as e:
                    console.print(f"[red]Failed to log in to Hugging Face: {e}[/red]")

        else:
            # Fallback to session token if something weird happens with key gen
            obj["api_key"] = access_token
            # We already saved access_token/refresh_token, so this is just about setting the api_key field
            # in config if they want to rely on the session token as their "key" (usually temporary)
            save_config(obj)
            console.print("[yellow]Logged in using session token (could not generate permanent key).[/yellow]")

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
        session = result.get("session", {})
        access_token = session.get("access_token")
        refresh_token = session.get("refresh_token")
        
        if access_token:
            obj["access_token"] = access_token
            if refresh_token:
                obj["refresh_token"] = refresh_token
            save_config(obj)
            
            mx.set_token(access_token, refresh_token=refresh_token)
            
            console.print(Panel("[bold green]Successfully logged in![/bold green]\nSession tokens saved to ~/.mechanex/config.json", title="Welcome"))
            
            # Hugging Face Login
            if click.confirm("\nDo you want to log in to Hugging Face now?", default=True):
                hf_token = click.prompt("Hugging Face Token", hide_input=True)
                try:
                    huggingface_hub.login(token=hf_token)
                    console.print("[bold green]Successfully logged in to Hugging Face![/bold green]")
                except Exception as e:
                    console.print(f"[red]Failed to log in to Hugging Face: {e}[/red]")
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

@main.command()
def balance():
    """Check your current account balance."""
    try:
        data = mx.get_balance()
        bal = data.get("balance", "0")
        console.print(Panel(f"Current Balance: [bold green]${bal}[/bold green]", title="Account Balance", border_style="green"))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@main.command()
def topup():
    """Purchase credits or subscribe to a plan."""
    config = load_config()
    if "access_token" not in config:
        console.print("[bold yellow]You must be logged in to make a payment.[/bold yellow] Please run `mechanex login`.")
        return

    try:
        products = mx.get_subscription_products()
        if not products:
            console.print("[yellow]No subscription products available at the moment.[/yellow]")
            return
            
        console.print("\n[bold cyan]Available Plans:[/bold cyan]")
        
        # Determine format of products. Expecting list of {name, prices: [{id, unit_amount, currency, etc}]}
        # But based on typical Stripe implementations, it might be Price objects or Products with prices.
        # Let's assume a simplified structure or try to handle what comes back.
        # If products is a list of Stripe Prices/Products
        
        options = []
        
        # Try to parse based on standard stripe structures or custom backend wrapper
        if isinstance(products, dict) and "data" in products:
            products = products["data"]
            
        table = Table(title="Select a Plan", show_lines=True)
        table.add_column("#", style="dim")
        table.add_column("Plan", style="bold")
        table.add_column("Price", style="green")
        table.add_column("Description")
        
        for idx, item in enumerate(products, 1):
            # Try to handle price object directly or product wrapper
            price_id = item.get("id")
            # If item represents a Product, it might have a price nested
            
            # Simple assumption based on user request "topup" usually implies one-off, but endpoint is "subscriptions"
            # Let's handle generic "name" and "price/amount"
            
            name = item.get("name", item.get("product_name", "Unknown Plan"))
            
            # Formatting price
            unit_amount = item.get("unit_amount")
            if unit_amount is None:
                unit_amount = 0
            amount = float(unit_amount) / 100.0

            if "price" in item: # Sometimes tailored responses put amount in 'price'
                 # Ensure this is also safe if needed, though usually strict
                 price_val = item["price"]
                 if price_val is not None:
                     amount = float(price_val)
            
            currency = item.get("currency", "usd").upper()
            price_str = f"{amount:.2f} {currency}"
            
            desc = item.get("description", "")
            
            # If we need to dig deeper into Stripe structure:
            if "nickname" in item and item["nickname"]:
                 name = item["nickname"]
            
            options.append((price_id, name, price_str))
            table.add_row(str(idx), name, price_str, desc)
            
        console.print(table)
        
        choice = click.prompt(f"\nSelect a plan (1-{len(options)})", type=int)
        
        if 1 <= choice <= len(options):
            selected_price_id, selected_name, selected_price = options[choice-1]
            console.print(f"\nInitiating checkout for [bold green]{selected_name}[/bold green] ({selected_price})...")
            
            checkout_res = mx.create_checkout_session(selected_price_id)
            url = checkout_res.get("url")
            
            if url:
                console.print(Panel(
                    f"Complete your payment here:\n\n[bold blue underline]{url}[/bold blue underline]",
                    title="Checkout Link",
                    border_style="green"
                ))
                click.launch(url)
            else:
                 console.print("[red]Failed to generate checkout URL.[/red]")
        else:
            console.print("[red]Invalid selection.[/red]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()

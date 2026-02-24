import click
import json
import os
import time
import asyncio
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
@click.option('--api-url', default=None, help='Override the backend API URL (e.g. http://localhost:3000).')
@click.option('--api-key', default=None, help='Axionic API key (overrides saved credentials).')
@click.pass_context
def main(ctx, api_url, api_key):
    """Mechanex CLI for managing your Axionic account and models."""
    config = load_config()

    # Override base URL if provided
    if api_url:
        mx.base_url = api_url

    # Load API Key: CLI flag takes priority over saved config
    if api_key:
        mx.set_key(api_key)
    elif "api_key" in config:
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

@main.group(name='manage-account')
def manage_account():
    """Manage your account settings."""
    pass

@manage_account.command(name='delete')
@click.confirmation_option(prompt='Are you sure you want to delete your account? This cannot be undone.')
def delete_account():
    """Permanently delete your account."""
    try:
        mx.delete_account()
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        console.print("[bold green]Account deleted successfully.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@manage_account.command(name='change-password')
@click.option('--new-password', prompt=True, hide_input=True, confirmation_prompt=True, help='New password.')
def change_password(new_password):
    """Change your password."""
    try:
        mx.change_password(new_password)
        console.print("[bold green]Password changed successfully.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@manage_account.command(name='change-email')
@click.option('--new-email', prompt='New Email', help='New email address.')
def change_email(new_email):
    """Change your email address."""
    try:
        mx.change_email(new_email)
        console.print("[bold green]Email change initiated. Please check your inbox for confirmation.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

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

@main.command(name='generate-data')
@click.option('--schemas-dir', required=True, help='Directory containing tool schema JSONs or .txt natural language files.')
@click.option('--teacher-provider', default='google', show_default=True, help='Teacher model provider (google, openai, anthropic).')
@click.option('--teacher-model', default='gemini-2.0-flash', show_default=True, help='Teacher model name.')
@click.option('--api-key', envvar='MECHANEX_API_KEY', default=None, help='API key for the teacher model.')
@click.option('--num-prompts', default=100, show_default=True, type=int, help='Number of seed prompts to generate.')
@click.option('--output-dir', default='output', show_default=True, help='Output directory where seeds.jsonl will be written.')
def generate_data(schemas_dir, teacher_provider, teacher_model, api_key, num_prompts, output_dir):
    """Generate seed prompts from schemas and save to seeds.jsonl."""
    from mechanex._training.utils.schema_loader import load_schemas
    from mechanex._training.seed_gen.data_pipeline.seed_generator import SeedGenerator, SeedGeneratorConfig
    from mechanex._training.seed_gen.api.gemini_client import GeminiConfig

    try:
        schemas = load_schemas(schemas_dir)
        if not schemas:
            console.print(f"[bold red]Error:[/bold red] No schemas found in '{schemas_dir}'.")
            return

        console.print(f"Loaded [bold cyan]{len(schemas)}[/bold cyan] schemas.")

        gemini_cfg = GeminiConfig(api_key=api_key or '', provider=teacher_provider, model_name=teacher_model)
        gen_cfg = SeedGeneratorConfig(num_prompts=num_prompts, output_format='jsonl')

        console.print(f"Generating [bold cyan]{num_prompts}[/bold cyan] seed prompts using {teacher_provider}/{teacher_model}...")
        generator = SeedGenerator(config=gen_cfg, gemini_config=gemini_cfg, tools=schemas)

        loop = asyncio.get_event_loop()
        batch = loop.run_until_complete(generator.generate())

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        seeds_file = output_path / 'seeds.jsonl'
        with open(seeds_file, 'w') as f:
            f.write(batch.to_jsonl())

        console.print(Panel(
            f"[bold green]Seed generation complete![/bold green]\n\n"
            f"Generated: [bold cyan]{batch.valid_count}[/bold cyan] valid seed prompts\n"
            f"Saved to:  [bold cyan]{seeds_file}[/bold cyan]",
            title="Generate Data",
            border_style="green"
        ))
        generator.print_stats()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command(name='train-sft')
@click.option('--schemas-dir', required=True, help='Directory containing tool schemas.')
@click.option('--data-dir', default='output', show_default=True, help='Directory containing seeds.jsonl from generate-data.')
@click.option('--num-trajectories', default=None, type=int, help='Number of prompt-response pairs to generate for training. Defaults to all seeds.')
@click.option('--model-name', default='Qwen/Qwen2.5-0.5B-Instruct', show_default=True, help='Base model to fine-tune.')
@click.option('--output-dir', default='sft_output', show_default=True, help='Directory for checkpoints and final model.')
@click.option('--teacher-provider', default=None, help='Teacher model provider (google, openai, anthropic).')
@click.option('--teacher-model', default=None, help='Teacher model name.')
@click.option('--api-key', envvar='MECHANEX_API_KEY', default=None, help='API key for the teacher model.')
@click.option('--learning-rate', default=2e-5, show_default=True, type=float, help='Learning rate.')
@click.option('--num-train-epochs', default=5, show_default=True, type=int, help='Number of training epochs.')
@click.option('--batch-size', default=2, show_default=True, type=int, help='Batch size per device.')
@click.option('--grad-acc', default=2, show_default=True, type=int, help='Gradient accumulation steps.')
@click.option('--warmup-ratio', default=0.1, show_default=True, type=float, help='Warmup ratio.')
@click.option('--lr-scheduler', default='cosine_with_min_lr', show_default=True, help='LR scheduler type.')
@click.option('--optim', default='adamw_8bit', show_default=True, help='Optimizer.')
@click.option('--weight-decay', default=0.01, show_default=True, type=float, help='Weight decay.')
@click.option('--eval-steps', default=100, show_default=True, type=int, help='Evaluate every N steps.')
@click.option('--save-steps', default=100, show_default=True, type=int, help='Save checkpoint every N steps.')
@click.option('--save-total-limit', default=25, show_default=True, type=int, help='Maximum checkpoints to keep.')
@click.option('--completion-only-loss', is_flag=True, default=True, help='Compute loss on completions only.')
@click.option('--wandb-project', default=None, help='Weights & Biases project name.')
@click.option('--wandb-run', default=None, help='Weights & Biases run name.')
@click.option('--report-to', default='wandb', show_default=True, help='Reporting backend (wandb, none).')
@click.option('--env-file', default=None, help='Python file containing a custom CRMEnvironment class.')
@click.option('--teacher-file', default=None, help='Python file containing a custom GeminiTeacher2 class.')
def train_sft(schemas_dir, data_dir, num_trajectories, model_name, output_dir, teacher_provider, teacher_model,
              api_key, learning_rate, num_train_epochs, batch_size, grad_acc, warmup_ratio, lr_scheduler,
              optim, weight_decay, eval_steps, save_steps, save_total_limit, completion_only_loss,
              wandb_project, wandb_run, report_to, env_file, teacher_file):
    """Run Supervised Fine-Tuning using seeds from a previous generate-data run."""
    from mechanex._training.utils.schema_loader import load_schemas
    from mechanex._training.training.sft import SFTTrainerModule
    from mechanex._training.utils.mock_env import AutoEnvironment
    from mechanex._training.utils.default_teacher import DefaultTeacher
    from mechanex._training.utils.loader import load_user_class

    try:
        import random

        if wandb_project:
            os.environ['WANDB_PROJECT'] = wandb_project
        if wandb_run:
            os.environ['WANDB_NAME'] = wandb_run

        schemas = load_schemas(schemas_dir)
        schemas_dicts = [s.to_json_schema() for s in schemas]

        if env_file:
            EnvClass = load_user_class(env_file, 'CRMEnvironment')
            env = EnvClass()
        else:
            env = AutoEnvironment(schemas_dicts)

        if teacher_file:
            TeacherClass = load_user_class(teacher_file, 'GeminiTeacher2')
            teacher = TeacherClass(schemas_dicts)
        else:
            if not teacher_provider:
                if os.getenv('GEMINI_API_KEY'):
                    teacher_provider = 'google'
                else:
                    console.print("[bold red]Error:[/bold red] --teacher-provider is required (or set GEMINI_API_KEY).")
                    return
            teacher = DefaultTeacher(provider_name=teacher_provider, schemas=schemas_dicts, model_name=teacher_model, api_key=api_key)

        trainer = SFTTrainerModule(model_name, env, teacher, schemas_dicts, output_dir)

        seeds_file = Path(data_dir) / 'seeds.jsonl'
        if not seeds_file.exists():
            console.print(f"[bold red]Error:[/bold red] seeds.jsonl not found in '{data_dir}'. Run generate-data first.")
            return

        with open(seeds_file) as f:
            seed_entries = [json.loads(line) for line in f if line.strip()]

        random.shuffle(seed_entries)

        if num_trajectories is not None:
            if num_trajectories > len(seed_entries):
                console.print(f"[yellow]Warning:[/yellow] --num-trajectories ({num_trajectories}) exceeds available seeds ({len(seed_entries)}). Using all seeds.")
            seed_entries = seed_entries[:num_trajectories]

        prompts = [entry['prompt'] for entry in seed_entries]
        console.print(f"Loaded [bold cyan]{len(prompts)}[/bold cyan] seed prompts from {seeds_file} (shuffled).")

        console.print("Generating trajectories...")
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(trainer.generate_training_data(prompts))

        console.print("Starting training...")
        dataset = trainer.format_dataset(data)

        train_args_patch = {
            'learning_rate': learning_rate,
            'num_train_epochs': num_train_epochs,
            'per_device_train_batch_size': batch_size,
            'gradient_accumulation_steps': grad_acc,
            'warmup_ratio': warmup_ratio,
            'lr_scheduler_type': lr_scheduler,
            'optim': optim,
            'weight_decay': weight_decay,
            'eval_steps': eval_steps,
            'save_steps': save_steps,
            'save_total_limit': save_total_limit,
            'completion_only_loss': completion_only_loss,
            'report_to': report_to,
        }

        trainer.train(dataset, train_args_patch=train_args_patch)

        console.print(Panel(
            f"[bold green]SFT training complete![/bold green]\n\n"
            f"Final model saved to: [bold cyan]{output_dir}/final_model[/bold cyan]",
            title="Train SFT",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command(name='train-rl')
@click.option('--model-name', required=True, help='Path to the SFT-trained model to start RL from.')
@click.option('--schemas-dir', required=True, help='Directory containing tool schemas.')
@click.option('--data-dir', default='output', show_default=True, help='Directory containing seeds.jsonl from generate-data.')
@click.option('--output-dir', default='rl_output', show_default=True, help='Directory for checkpoints and final model.')
@click.option('--teacher-provider', default='google', show_default=True, help='Teacher model provider for ARA reward (google, openai, anthropic).')
@click.option('--teacher-model', default='gemini-2.0-flash', show_default=True, help='Teacher model name for ARA reward scoring.')
@click.option('--api-key', envvar='MECHANEX_API_KEY', default=None, help='API key for the teacher/reward model.')
@click.option('--learning-rate', default=5e-6, show_default=True, type=float, help='Learning rate.')
@click.option('--num-train-epochs', default=1, show_default=True, type=int, help='Number of RL training epochs.')
@click.option('--batch-size', default=1, show_default=True, type=int, help='Batch size per device.')
@click.option('--grad-acc', default=8, show_default=True, type=int, help='Gradient accumulation steps.')
@click.option('--num-generations', default=4, show_default=True, type=int, help='Number of completions per prompt for GRPO.')
@click.option('--max-prompt-length', default=512, show_default=True, type=int, help='Maximum prompt length in tokens.')
@click.option('--max-completion-length', default=256, show_default=True, type=int, help='Maximum completion length in tokens.')
@click.option('--report-to', default='none', show_default=True, help='Reporting backend (wandb, none).')
def train_rl(model_name, schemas_dir, data_dir, output_dir, teacher_provider, teacher_model, api_key,
             learning_rate, num_train_epochs, batch_size, grad_acc, num_generations,
             max_prompt_length, max_completion_length, report_to):
    """Run Reinforcement Learning (GRPO) using seeds from a previous generate-data run."""
    from mechanex._training.utils.schema_loader import load_schemas
    from mechanex._training.training.rl import RLTrainerModule
    from mechanex._training.utils.default_teacher import DefaultTeacher
    from datasets import Dataset

    try:
        import random
        from tqdm import tqdm

        schemas = load_schemas(schemas_dir)
        schemas_dicts = [s.to_json_schema() for s in schemas]

        ara_config = {
            'teacher_provider': teacher_provider,
            'teacher_model': teacher_model,
            'teacher_api_key': api_key
        }

        trainer = RLTrainerModule(model_name, schemas_dicts, ara_config=ara_config, output_dir=output_dir)

        seeds_file = Path(data_dir) / 'seeds.jsonl'
        if not seeds_file.exists():
            console.print(f"[bold red]Error:[/bold red] seeds.jsonl not found in '{data_dir}'. Run generate-data first.")
            return

        with open(seeds_file) as f:
            data = [json.loads(line) for line in f if line.strip()]

        random.shuffle(data)
        console.print(f"Loaded [bold cyan]{len(data)}[/bold cyan] seed prompts from {seeds_file} (shuffled).")

        dataset = Dataset.from_list(data)

        sample_size = min(len(dataset), 5)
        needs_generation = any(
            'response' not in dataset[i] or not dataset[i].get('response')
            for i in range(sample_size)
        )

        if needs_generation:
            console.print("Generating Teacher Trajectories (Ground Truth) for RL...")
            if not api_key:
                if os.environ.get('GEMINI_API_KEY'):
                    api_key = os.environ.get('GEMINI_API_KEY')
                else:
                    console.print("[yellow]Warning:[/yellow] No API key for generating teacher traces. RL may fail if ground truth is missing.")

            teacher = DefaultTeacher(provider_name=teacher_provider, schemas=schemas_dicts, model_name=teacher_model, api_key=api_key)

            dataset_list = list(dataset)
            prompts = [x['prompt'] for x in dataset_list]

            loop = asyncio.get_event_loop()

            async def aug_data():
                augmented = []
                for i, p in enumerate(tqdm(prompts, desc="Generating Trajectories")):
                    trace = await teacher.generate_trace(p)
                    item = dataset_list[i].copy()
                    item['response'] = trace['raw'] if trace['success'] and trace['tool_call'] else ''
                    augmented.append(item)
                return augmented

            dataset_list = loop.run_until_complete(aug_data())
            dataset = Dataset.from_list(dataset_list)

            augmented_path = Path(data_dir) / 'seeds_with_gt.jsonl'
            console.print(f"Saving augmented ground truth to [cyan]{augmented_path}[/cyan]")
            with open(augmented_path, 'w') as f:
                for item in dataset_list:
                    f.write(json.dumps(item) + '\n')

        console.print("Formatting dataset and starting RL training...")
        dataset = trainer.format_dataset(dataset)

        train_args = {
            'learning_rate': learning_rate,
            'num_train_epochs': num_train_epochs,
            'per_device_train_batch_size': batch_size,
            'gradient_accumulation_steps': grad_acc,
            'num_generations': num_generations,
            'max_prompt_length': max_prompt_length,
            'max_completion_length': max_completion_length,
            'report_to': report_to,
        }

        trainer.train(dataset, train_args_patch=train_args)

        console.print(Panel(
            f"[bold green]RL training complete![/bold green]\n\n"
            f"Final model saved to: [bold cyan]{output_dir}/final_rl_model[/bold cyan]",
            title="Train RL",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command(name='run-eval')
@click.option('--job-id', required=True, help='Training job execution name (returned when training was submitted).')
def run_eval(job_id):
    """Check the status of a training job."""
    try:
        result = mx.check_training_job(job_id)
        status = result.get("status", "UNKNOWN")
        start_time = result.get("start_time") or "N/A"
        completion_time = result.get("completion_time") or "N/A"
        log_uri = result.get("log_uri") or ""

        color = {"SUCCEEDED": "green", "FAILED": "red", "RUNNING": "yellow", "PENDING": "cyan"}.get(status, "white")
        info = (
            f"Status: [bold {color}]{status}[/bold {color}]\n"
            f"Started:   {start_time}\n"
            f"Completed: {completion_time}"
        )
        if log_uri:
            info += f"\nLogs: [blue]{log_uri}[/blue]"

        console.print(Panel(info, title=f"Job: {job_id}", border_style=color))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command(name='deploy')
@click.option('--job-id', default=None, help='Training job execution name to check upload status.')
def deploy(job_id):
    """Show deployment info. Models are auto-uploaded to HuggingFace after training completes."""
    msg = (
        "[bold green]Automatic deployment is handled by the training pipeline.[/bold green]\n\n"
        "After training completes, your model is automatically uploaded to HuggingFace.\n"
        "Use [bold cyan]mechanex run-eval --job-id <execution_name>[/bold cyan] to check job status."
    )
    if job_id:
        try:
            result = mx.check_training_job(job_id)
            status = result.get("status", "UNKNOWN")
            color = {"SUCCEEDED": "green", "FAILED": "red", "RUNNING": "yellow", "PENDING": "cyan"}.get(status, "white")
            msg += f"\n\nJob [bold cyan]{job_id}[/bold cyan] status: [bold {color}]{status}[/bold {color}]"
        except Exception as e:
            msg += f"\n\n[yellow]Could not fetch job status: {e}[/yellow]"
    console.print(Panel(msg, title="Deploy", border_style="green"))


if __name__ == "__main__":
    main()

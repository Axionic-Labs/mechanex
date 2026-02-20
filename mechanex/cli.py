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
    if "api_key" in config:
        mx.set_key(config["api_key"])
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
        
        # Check if signup returned a session token (some backends do this)
        # Handle case where signup_result is None
        if signup_result is None:
            signup_result = {}

        session = signup_result.get("session", {})

        if session is not None:
            session_token = session.get("access_token")
            console.print("Authenticated via signup response.")
            mx.set_key(session_token)
        else:
            # 2. Auto-Login with retry
            console.print("Logging in to generate credentials...")
            # Wait a moment for DB propagation if needed
            time.sleep(1) 
            
            try:
                login_result = mx.login(email, password)
                session_token = login_result.get("session", {}).get("access_token")
            except Exception:
                # Retry once
                console.print("Retrying login...")
                time.sleep(2)
                login_result = mx.login(email, password)
                session_token = login_result.get("session", {}).get("access_token")
        
        if not session_token:
            console.print("[yellow]Signup successful, but could not log in automatically.[/yellow]")
            return

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
            obj["api_key"] = session_token
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
        token = result.get("session", {}).get("access_token")
        if token:
            obj["api_key"] = token
            save_config(obj)
            console.print(Panel("[bold green]Successfully logged in![/bold green]\nAPI key saved to ~/.mechanex/config.json", title="Welcome"))
            
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

# ---------------------------------------------------------------------------
# Training pipeline commands
# ---------------------------------------------------------------------------

@main.command()
@click.option('--num-seeds', default=10, show_default=True, help='Number of prompts to generate.')
@click.option('--output-file', default='seeds.jsonl', show_default=True, help='Output file name stored in tenant directory.')
@click.option('--topic', default='customer relationship management', show_default=True, help='Broad topic for generation.')
@click.option('--teacher-provider', default='google', show_default=True, help='LLM provider for the teacher (e.g. "google").')
@click.option('--teacher-model', default='gemini-2.0-flash', show_default=True, help='Teacher model name.')
@click.option('--api-key', default=None, help='Optional API key for the teacher provider.')
@click.option('--teacher-file', default=None, help='Optional path to a custom teacher implementation.')
def generate_data(num_seeds, output_file, topic, teacher_provider, teacher_model, api_key, teacher_file):
    """Generate seed prompt data using a teacher model."""
    try:
        result = mx.training.generate_data(
            num_seeds=num_seeds,
            output_file=output_file,
            topic=topic,
            teacher_provider=teacher_provider,
            teacher_model=teacher_model,
            api_key=api_key,
            teacher_file=teacher_file,
        )
        console.print(result)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command()
@click.option('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct', show_default=True, help='HuggingFace model ID for the base model.')
@click.option('--prompts-file', default='seeds.jsonl', show_default=True, help='Path to seed prompts file (relative to tenant directory).')
@click.option('--schemas-dir', default='tool_schemas', show_default=True, help='Directory containing tool schemas (relative to tenant directory).')
@click.option('--output-dir', default='sft_output', show_default=True, help='Output directory for the trained model (relative to tenant directory).')
@click.option('--teacher-model', default='gemini-2.0-flash', show_default=True, help='Teacher model for trajectory generation.')
@click.option('--teacher-provider', default='google', show_default=True, help='LLM provider for the teacher.')
@click.option('--epochs', default=3, show_default=True, help='Number of training epochs.')
@click.option('--batch-size', default=4, show_default=True, help='Training batch size.')
@click.option('--learning-rate', default=2e-5, show_default=True, help='Training learning rate.')
@click.option('--gradient-accumulation-steps', default=8, show_default=True, help='Number of gradient accumulation steps.')
@click.option('--use-peft/--no-peft', default=True, show_default=True, help='Use Parameter Efficient Fine-Tuning (LoRA).')
def train_sft(base_model, prompts_file, schemas_dir, output_dir, teacher_model, teacher_provider,
              epochs, batch_size, learning_rate, gradient_accumulation_steps, use_peft):
    """Run Supervised Fine-Tuning (SFT) on a base model."""
    try:
        result = mx.training.train_sft(
            base_model=base_model,
            prompts_file=prompts_file,
            schemas_dir=schemas_dir,
            output_dir=output_dir,
            teacher_model=teacher_model,
            teacher_provider=teacher_provider,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_peft=use_peft,
        )
        console.print(result)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command()
@click.option('--model-path', default='sft_output/best_checkpoint', show_default=True, help='Path to the SFT-trained model (relative to tenant directory).')
@click.option('--output-dir', default='rl_output', show_default=True, help='Output directory for the RL-trained model (relative to tenant directory).')
@click.option('--prompts-file', default='seeds.jsonl', show_default=True, help='Path to prompts file for RL training (relative to tenant directory).')
@click.option('--schemas-dir', default='tool_schemas', show_default=True, help='Directory containing tool schemas (relative to tenant directory).')
@click.option('--teacher-model', default='gemini-2.0-flash', show_default=True, help='Teacher model for rewards/eval.')
@click.option('--teacher-provider', default='google', show_default=True, help='LLM provider for the teacher.')
@click.option('--num-train-epochs', default=1, show_default=True, help='Number of RL training epochs.')
@click.option('--batch-size', default=1, show_default=True, help='Training batch size.')
@click.option('--gradient-accumulation-steps', default=8, show_default=True, help='Number of gradient accumulation steps.')
@click.option('--num-generations', default=4, show_default=True, help='Number of generations per prompt for GRPO.')
def train_rl(model_path, output_dir, prompts_file, schemas_dir, teacher_model, teacher_provider,
             num_train_epochs, batch_size, gradient_accumulation_steps, num_generations):
    """Run Reinforcement Learning (GRPO) training on an SFT checkpoint."""
    try:
        result = mx.training.train_rl(
            model_name_or_path=model_path,
            output_dir=output_dir,
            prompts_file=prompts_file,
            schemas_dir=schemas_dir,
            teacher_model=teacher_model,
            teacher_provider=teacher_provider,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
        )
        console.print(result)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command()
@click.argument('model-name')
@click.option('--num-eval-samples', default=50, show_default=True, help='Number of fresh samples to generate for evaluation.')
@click.option('--output-dir', default='eval_output', show_default=True, help='Output directory for evaluation results (relative to tenant directory).')
@click.option('--schemas-dir', default='tool_schemas', show_default=True, help='Directory containing tool schemas (relative to tenant directory).')
@click.option('--teacher-model', default='gemini-2.0-flash', show_default=True, help='Teacher model for reference trajectories.')
@click.option('--teacher-provider', default='google', show_default=True, help='LLM provider for the teacher.')
def run_eval(model_name, num_eval_samples, output_dir, schemas_dir, teacher_model, teacher_provider):
    """Evaluate a trained model. MODEL_NAME is the path to the model (relative to tenant directory)."""
    try:
        result = mx.training.run_eval(
            model_name=model_name,
            num_eval_samples=num_eval_samples,
            output_dir=output_dir,
            schemas_dir=schemas_dir,
            teacher_model=teacher_model,
            teacher_provider=teacher_provider,
        )
        console.print(result)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command()
@click.argument('model-path')
@click.option('--extra-args', multiple=True, help='Additional arguments to pass to vLLM serve (repeatable).')
def deploy(model_path, extra_args):
    """Deploy a trained model using vLLM serve. MODEL_PATH is relative to tenant directory."""
    try:
        result = mx.training.deploy(
            model_path=model_path,
            extra_args=list(extra_args) if extra_args else None,
        )
        console.print(result)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@main.command()
@click.argument('schema_files', nargs=-1, required=True)
def upload_schemas(schema_files):
    """Upload tool schema JSON files to your tenant directory.

    SCHEMA_FILES: One or more local paths to JSON schema files.
    """
    try:
        result = mx.training.upload_schemas(list(schema_files))
        console.print(result)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()

from typing import Callable, Optional, TypeVar, Union

import click
import typer

T = TypeVar("T")


def find_command_info(typer_instance: typer.Typer, callback: Callable) -> Optional[typer.models.CommandInfo]:
    """Return a CommandInfo that is contained within a Typer instance."""
    for command_info in typer_instance.registered_commands:
        if command_info.callback == callback:
            return command_info
    for group in typer_instance.registered_groups:
        command_info = find_command_info(group.typer_instance, callback)
        if command_info:
            return command_info
    return None


def find_typer_info(typer_instance: typer.Typer, callback: Callable) -> Optional[typer.models.TyperInfo]:
    """Return a TyperInfo that is contained within a Typer instance."""
    if typer_instance.registered_callback.callback == callback:
        return typer_instance.registered_callback
    for group in typer_instance.registered_groups:
        typer_info = find_typer_info(group.typer_instance, callback)
        if typer_info:
            return typer_info
    return None


def callback_to_click_command(
    typer_instance: typer.Typer,
    callback: Union[Callable[..., T], click.Command],
) -> Union[Callable[..., T], click.Command]:
    """
    Return the click.Command object for a given callable, if it is registered under the given Typer instance.

    If the callback is not a registered command, just returns the callback.
    """
    command_info = find_command_info(typer_instance, callback)
    if command_info:
        callback = typer.main.get_command_from_info(command_info)
    else:
        typer_info = find_typer_info(typer_instance, callback)
        if typer_info:
            typer_info.typer_instance = typer_instance
            callback = typer.main.get_group_from_info(typer_info)
    return callback


def invoke(typer_instance: typer.Typer, callback: Callable[..., T], *args, **kwargs) -> T:
    """
    Invoke a callable that is a registered command or subcommand of the typer_instance.

    See `click.Context.invoke`.
    """
    return click.get_current_context().invoke(callback_to_click_command(typer_instance, callback), *args, **kwargs)


def forward(typer_instance: typer.Typer, callback: Callable[..., T], *args, **kwargs) -> T:
    """
    Forward a callable that is a registered command or subcommand of the typer_instance.

    See `click.Context.forward`.
    """
    return click.get_current_context().forward(callback_to_click_command(typer_instance, callback), *args, **kwargs)

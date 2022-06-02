import functools
import hashlib
import io
import multiprocessing
import subprocess  # noqa: S404
import urllib.request
import zipfile
from collections import OrderedDict
from importlib import resources
from pathlib import Path

from loguru import logger

from ..exceptions import BuckyException
from .import_by_name import import_by_name
from .working_directory import working_directory


class BuckySyncException(BuckyException):
    pass


def _locate_included_data():
    """locate the base_config package that shipped with bucky (it's likely in site-packages)."""
    with resources.path("bucky", "included_data") as inc_data_path:
        return Path(inc_data_path)


def _hash_file_obj(obj):
    return hashlib.sha256(obj).hexdigest()


def _unzip_file_obj_to_dir(f, output_dir=None):
    with zipfile.ZipFile(f) as z:
        z.extractall(output_dir)


# def _get_file_from_url(url, ssl_no_verify=False):
#    return
def _write_filelike(src, dest, buffer_size=16384):
    while True:
        buffer = src.read(buffer_size)
        if not buffer:
            break
        dest.write(buffer)


def _exec_shell_cmd(cmd, cwd=None):
    # with subprocess.Popen(cmd.split(), capture_output=True, test=True, cwd=cwd)
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True, cwd=cwd)  # noqa: S603
        logger.debug("Successfully executed shell cmd '{}':\nstdout:\n{}stderr:\n{}", cmd, result.stdout, result.stderr)
    except subprocess.CalledProcessError as ex:
        logger.exception(
            "Error executing shell cmd '{}':\nreturn code:\n{}\nstdout:\n{}stderr:\n{}",
            ex.cmd,
            ex.returncode,
            ex.stdout,
            ex.stderr,
        )
        raise BuckySyncException from ex


def _git_clone(url, local_name, abs_path, bare=False, depth=1, tag=None):
    """Updates a git repository given its path.

    Parameters
    ----------
    abs_path : str
        Abs path location of repository to update
    """
    # build git command
    git_command = "git clone --progress "
    if depth is not None:
        git_command += f"--depth={depth} "
    if bare:
        git_command += "--bare "
    if tag is not None:
        git_command += f"--branch {tag} "
    git_command += f"{url} "
    git_command += local_name

    logger.debug("Cloning git repo to {} with cmd: '{}'", abs_path, git_command)

    # run git subprocess
    _exec_shell_cmd(git_command, cwd=abs_path)


def _git_pull(abs_path, rebase=True):
    # build git command
    git_command = "git pull --progress "
    if rebase:
        git_command += "--rebase "

    logger.debug("Pulling git repo at {} cmd: '{}'", abs_path, git_command)

    # run git subprocess
    _exec_shell_cmd(git_command, cwd=abs_path)


def process_datasources(data_sources, data_dir, ssl_no_verify=False, n_jobs=None):

    raw_data_dir = data_dir / "raw"
    raw_data_dir.mkdir(exist_ok=True, parents=True)

    _process_one = functools.partial(_process_one_datasource, raw_data_dir=raw_data_dir)

    # copy included data over
    included_data_path = _locate_included_data()
    included_data_link = raw_data_dir / "included_data"
    if not included_data_link.exists():
        included_data_link.symlink_to(included_data_path, target_is_directory=True)

    priority_sources = OrderedDict({source["priority"]: source for source in data_sources if "priority" in source})

    normal_sources = [source for source in data_sources if "priority" not in source]

    # Process priority sources serially in order
    for source_cfg in priority_sources.values():
        _process_one(source_cfg)

    # Process remaining data sources (potentially in parallel)
    if n_jobs is None:
        for source_cfg in normal_sources:
            _process_one(source_cfg)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        res = pool.map_async(_process_one, normal_sources, chunksize=1)
        # TODO add some polling on the map result object to update log status?
        while not res.ready():
            logger.info("Still working on {}/{} data_sources", res._number_left, len(normal_sources))
            res.wait(timeout=10)
        # TODO need to check for exceptions with res.successful()
        pool.close()
        pool.join()


def _process_one_datasource(source_cfg, raw_data_dir):
    f_path = raw_data_dir / source_cfg["name"]

    if source_cfg["type"] == "git":

        if (f_path / ".git").exists():
            logger.info("Updating git repo {}...", source_cfg["name"])
            _git_pull(abs_path=raw_data_dir / source_cfg["name"])
            # TODO if stdout contains "Already up to date." skip post?

        elif f_path.exists():
            logger.error("Cannot sync git repo {}! (is it a bare repo?)", source_cfg["name"])
            raise RuntimeError(f"Cannot sync git repo {source_cfg['name']}")

        else:
            logger.info("Git repo {}, not found in data_dir. Cloning...", source_cfg["name"])
            _git_clone(source_cfg["url"], local_name=source_cfg["name"], abs_path=raw_data_dir)

    if source_cfg["type"] == "http":

        # this zip checking is dodgey, need to just move it to a postprocessing step...
        is_zip = False if "unzip" not in source_cfg else bool(source_cfg["unzip"])
        ext = ".zip" if is_zip else source_cfg["ext"]
        raw_file = f_path / (source_cfg["name"] + ext)

        to_download = True
        if raw_file.exists() and "hash" in source_cfg:  # noqa: SIM102
            if _hash_file_obj(raw_file.read_bytes()) == source_cfg["hash"]:
                logger.info("Skipping download of file {} (existing hash matched cfg)", source_cfg["name"])
                to_download = False

        if to_download:
            f_path.mkdir(exist_ok=True)

            logger.info("Downloading {}...", source_cfg["url"])
            with urllib.request.urlopen(source_cfg["url"]) as tmp_file:  # noqa: S310
                f_obj = io.BytesIO(tmp_file.read())

            f_dat = f_obj.read()
            f_hash = _hash_file_obj(f_dat)
            logger.info("{} hash: {}", source_cfg["name"], f_hash)
            if "hash" in source_cfg:
                if f_hash == source_cfg["hash"]:
                    logger.info("Downloaded file has correct hash")
                else:
                    logger.error("Hash mismatch between cfg and downloaded file! Expected: {}", source_cfg["hash"])

            logger.info("Writing {} to {}", source_cfg["name"], f_path)
            raw_file.write_bytes(f_dat)

            if "unzip" in source_cfg:  # TODO move to a postprocessing step?
                logger.info("Unzipping {}", source_cfg["name"])
                _unzip_file_obj_to_dir(raw_file, f_path)

    if "postprocess" in source_cfg:

        for post_step in source_cfg["postprocess"]:

            logger.info("Postprocessing {}: {}", source_cfg["name"], post_step["desc"])
            # TODO skip if we already have a hash match

            with working_directory(f_path):

                if "check_hash" in post_step:  # noqa: SIM102
                    if post_step["check_hash"]["file"].exists():
                        if (
                            _hash_file_obj(post_step["check_hash"]["file"].read_bytes())
                            == post_step["check_hash"]["hash"]
                        ):
                            logger.info("Hash matched for output of {}, skipping.", post_step["desc"])
                            continue
                        else:
                            logger.warning(
                                "Postprocessing '{}', found file {} but there is a hash mismatch; rerunning {}",
                                post_step["desc"],
                                post_step["check_hash"]["file"],
                                post_step["func"],
                            )

                func = import_by_name(post_step["func"])
                # TODO check for errors in the sig of the called function
                func(**(post_step["args"]))

                if "check_hash" in post_step:

                    f_hash = _hash_file_obj(post_step["check_hash"]["file"].read_bytes())

                    if f_hash == post_step["check_hash"]["hash"]:
                        logger.info("Correct hash for {}.", post_step["check_hash"]["file"])
                    else:
                        logger.error(
                            "Hash mismatch for {}:\nexpected: {}\nactual: {}",
                            post_step["check_hash"]["file"].resolve(),
                            post_step["check_hash"]["hash"],
                            f_hash,
                        )

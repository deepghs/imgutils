from hfutils.operate import get_hf_client


def get_file_timestamp(repo_id: str, filename: str, repo_type: str = 'model'):
    hf_client = get_hf_client()
    return hf_client.get_paths_info(
        repo_id=repo_id,
        repo_type=repo_type,
        paths=[filename],
        expand=True
    )[0].last_commit.date.timestamp()

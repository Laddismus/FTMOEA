from research_lab.backend.core.job_runner import InMemoryJobRunner


def test_job_runner_stub_lifecycle() -> None:
    runner = InMemoryJobRunner()

    job_id = runner.submit_job("test", {"x": 1})
    status = runner.get_status(job_id)

    assert status["job_id"] == job_id
    assert status["status"] == "queued"

    jobs = runner.list_jobs()
    assert any(job["job_id"] == job_id for job in jobs)

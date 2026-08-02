"""Regression tests for the clean company-exclusive FL partition."""

import importlib.util
from pathlib import Path


PARTITION_PATH = (
    Path(__file__).resolve().parents[1]
    / "flgo"
    / "benchmark"
    / "partition.py"
)
SPEC = importlib.util.spec_from_file_location(
    "fede_partition_standalone", PARTITION_PATH
)
PARTITION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARTITION)
CompanyPartitioner = PARTITION.CompanyPartitioner
PaperFiveCompanyPartitioner = PARTITION.PaperFiveCompanyPartitioner


class DummyCompanyDataset:
    def __init__(self, companies):
        self.id = companies

    def __len__(self):
        return len(self.id)


def test_company_partition_is_exclusive_complete_and_deterministic():
    companies = (
        ["PEPSICO"] * 11
        + ["BOEING"] * 9
        + ["PFIZER"] * 7
        + ["INTEL"] * 5
        + ["WALMART"] * 3
        + ["AES"] * 2
    )
    data = DummyCompanyDataset(companies)

    first = CompanyPartitioner(num_clients=3)
    second = CompanyPartitioner(num_clients=3)
    first_indices = first(data)
    second_indices = second(data)

    assert first_indices == second_indices
    assert sorted(index for client in first_indices for index in client) == list(
        range(len(data))
    )
    assert all(first_indices)

    owners = {}
    for client_id, indices in enumerate(first_indices):
        for index in indices:
            owners.setdefault(companies[index], set()).add(client_id)
    assert all(len(client_ids) == 1 for client_ids in owners.values())
    assert first.company_to_client == {
        company: next(iter(client_ids))
        for company, client_ids in owners.items()
    }


def test_company_partition_normalizes_equivalent_spellings():
    data = DummyCompanyDataset(
        ["Johnson & Johnson", "JOHNSONJOHNSON", "A-B", "AB", "C"]
    )
    partitioner = CompanyPartitioner(num_clients=3)
    partitioner(data)

    assert partitioner.company_to_client["JOHNSONJOHNSON"] in range(3)
    assert "AB" in partitioner.company_to_client
    assert len(partitioner.company_to_client) == 3


def test_company_partition_rejects_empty_clients():
    data = DummyCompanyDataset(["A", "A", "B"])
    try:
        CompanyPartitioner(num_clients=3)(data)
    except ValueError as exc:
        assert "cannot populate" in str(exc)
    else:
        raise AssertionError("expected a ValueError for more clients than companies")


def test_paper_five_partition_uses_reported_client_order():
    data = DummyCompanyDataset(
        [
            "PEPSICO",
            "AES",
            "PG",
            "BOEING",
            "ACTIVISIONBLIZZARD",
            "PEPSICO",
        ]
    )
    partitioner = PaperFiveCompanyPartitioner(num_clients=5)
    partitions = partitioner(data)

    assert partitioner.client_companies == [
        ["AES"],
        ["BOEING"],
        ["ACTIVISIONBLIZZARD"],
        ["PG"],
        ["PEPSICO"],
    ]
    assert [len(indices) for indices in partitions] == [1, 1, 1, 1, 2]


def test_paper_five_partition_rejects_expanded_roster():
    data = DummyCompanyDataset(
        ["AES", "BOEING", "ACTIVISIONBLIZZARD", "PG", "PEPSICO", "AMCOR"]
    )
    try:
        PaperFiveCompanyPartitioner(num_clients=5)(data)
    except ValueError as exc:
        assert "company roster mismatch" in str(exc)
    else:
        raise AssertionError("expected expanded company roster to be rejected")

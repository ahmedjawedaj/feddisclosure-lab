"""Regression tests for adversarial experiment-ID and policy switching."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import httpx
import pytest

from feddisclosure.analytics.policy import evaluate_at_site
from feddisclosure.common.models import AggregateRequest
from feddisclosure.data.synthetic import InstitutionSpec, initialize_database
from services.client.app import ClientSettings, create_app
from services.gateway.app import GatewaySettings, create_app as create_gateway


def test_experiment_id_reset_exposes_history_but_not_principal_history(tmp_path):
    db = str(tmp_path / 'site.sqlite')
    initialize_database(db, InstitutionSpec('site-one', 40, 1))

    def ask(policy, query, exp):
        return evaluate_at_site(db, AggregateRequest(policy=policy, query_id=query, experiment_id=exp), 5)

    assert ask('history', 'all_records', 'a').count == 40
    assert ask('history', 'negative_records', 'b').count == 39  # bypass
    assert ask('principal_history', 'all_records', 'a').count == 40
    assert ask('principal_history', 'all_records', 'c').count == 40  # harmless repeat
    blocked = ask('principal_history', 'negative_records', 'b')
    assert not blocked.approved and blocked.count is None
    assert blocked.reason == 'near_duplicate_complement'
    assert ask('principal_history', 'all_records', 'd').approved
    with sqlite3.connect(db) as conn:
        audit = conn.execute('SELECT COUNT(*) FROM query_audit').fetchone()[0]
        assert audit == 6


def test_cross_experiment_race_releases_at_most_one_differencing_query(tmp_path):
    db = str(tmp_path / 'site.sqlite')
    initialize_database(db, InstitutionSpec('site-one', 40, 1))

    def ask(query, exp):
        return evaluate_at_site(db, AggregateRequest(policy='principal_history', query_id=query, experiment_id=exp), 5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ask, 'all_records', 'first'), pool.submit(ask, 'negative_records', 'second')]
        answers = [f.result(timeout=10) for f in futures]
    assert sum(a.approved for a in answers) == 1
    assert sum(not a.approved for a in answers) == 1


def test_enforced_site_rejects_policy_switch_and_sticky_downgrade(tmp_path):
    db = str(tmp_path / 'site.sqlite')
    settings = ClientSettings('site-one', db, 40, 1, 5, 'secret', enforced_policy='principal_history')
    app = create_app(settings)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://site') as client:
            hdr = {'X-Internal-Token': 'secret'}
            r = await client.post('/v1/aggregate', headers=hdr, json={
                'query_id': 'all_records', 'policy': 'principal_history', 'experiment_id': 'a'})
            assert r.status_code == 200
            r = await client.post('/v1/aggregate', headers=hdr, json={
                'query_id': 'negative_records', 'policy': 'principal_history', 'experiment_id': 'new'})
            assert r.status_code == 403 and 'count' not in r.json()
            r = await client.post('/v1/aggregate', headers=hdr, json={
                'query_id': 'negative_records', 'policy': 'naive', 'experiment_id': 'new'})
            assert r.status_code == 403 and 'count' not in r.json()
    asyncio.run(check())

    with pytest.raises(ValueError, match='immutable'):
        create_app(ClientSettings('site-one', db, 40, 1, 5, 'secret'))
    with pytest.raises(ValueError, match='immutable'):
        create_app(ClientSettings('site-one', db, 40, 1, 5, 'secret', enforced_policy='history'))
    assert create_app(settings)  # permitted restart with same locked mode


def test_protected_mode_refuses_prior_naive_release(tmp_path):
    db = str(tmp_path / 'prior.sqlite')
    initialize_database(db, InstitutionSpec('site-one', 40, 1))
    evaluate_at_site(db, AggregateRequest(policy='naive', query_id='all_records', experiment_id='old'), 5)
    with pytest.raises(ValueError, match='fresh site'):
        create_app(ClientSettings('site-one', db, 40, 1, 5, 'secret', enforced_policy='principal_history'))


def test_gateway_policy_switch_prevented_before_any_site_request(monkeypatch):
    app = create_gateway(GatewaySettings(('http://one', 'http://two'), 'secret', 'research',
                                         enforced_policy='principal_history'))

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway') as c:
            r = await c.post('/v1/federated-count', headers={'X-Research-Key': 'research'},
                             json={'policy': 'naive', 'query_id': 'all_records', 'experiment_id': 'x'})
            assert r.status_code == 403
    asyncio.run(check())


def test_unknown_policy_and_dp_conflict_fail_closed(tmp_path):
    base = dict(institution='site-one', db_path=str(tmp_path / 'site.sqlite'), rows=40,
                positive=1, min_cell=5, internal_token='secret')
    with pytest.raises(ValueError, match='Unknown enforced'):
        create_app(ClientSettings(**base, enforced_policy='unsafe'))
    with pytest.raises(ValueError, match='cannot both'):
        create_app(ClientSettings(**base, enforced_policy='principal_history', dp_enabled=True))

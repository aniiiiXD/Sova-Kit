'use strict';

const byId = id => document.getElementById(id);
const endpoint = '/api/switchup/projects';
const view = {project: null, selected: null, round: null, request: 0};

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

async function request(path, body) {
  const response = await fetch(path, body === undefined ? undefined : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || 'The workspace could not complete this request.');
  return data;
}

function notify(message, error = false) {
  const status = byId('up-message');
  status.textContent = message;
  status.className = error ? 'error-panel' : 'result-note';
  status.hidden = !message;
}

function roundButton(summary) {
  const button = node('button', undefined, 'job-item up-round-button');
  button.type = 'button';
  if (summary.id === view.round) button.classList.add('selected');
  const details = node('small');
  details.append(node('span', summary.id.slice(0, 8)), node('span', `Generation ${summary.generation}`));
  button.append(node('strong', summary.status.replaceAll('_', ' ')), details);
  button.addEventListener('click', () => selectRound(summary.id));
  return button;
}

function renderProject(project) {
  view.project = project;
  byId('up-name').textContent = project.name;
  byId('up-provider').textContent = `${project.provider} / ${project.model}`;
  byId('up-generation').textContent = project.generation;
  byId('up-recipe').textContent = project.active_recipe_id.slice(0, 12);
  byId('up-recipe').title = project.active_recipe_id;
  byId('up-events').textContent = project.event_count;
  byId('up-cases').textContent = project.case_count;
  byId('up-rollback').hidden = !project.can_rollback;
  byId('up-rounds').replaceChildren(...(project.rounds.length ? project.rounds.map(roundButton) : [
    node('p', 'Prepare an improvement round after pinning a replay suite.', 'empty-state'),
  ]));
  byId('up-project').hidden = false;
}

function renderPairs(cases) {
  const wrap = node('div', undefined, 'up-table-wrap');
  const table = node('table', undefined, 'up-table');
  const headers = node('tr');
  for (const name of ['Case', 'Split', 'Active', 'Candidate', 'State / first tool difference']) {
    const cell = node('th', name);
    cell.scope = 'col';
    headers.append(cell);
  }
  const head = node('thead');
  head.append(headers);
  const body = node('tbody');
  for (const pair of cases) {
    const row = node('tr');
    row.append(node('td', pair.case_id.slice(0, 10)), node('td', pair.split));
    for (const value of [pair.baseline_verdict, pair.candidate_verdict]) {
      const cell = node('td');
      cell.append(node('span', value || 'unobserved', `verdict ${value || 'uncertain'}`));
      row.append(cell);
    }
    const outcome = node('td');
    if (pair.final_state_matches !== null) {
      outcome.append(node('span', pair.final_state_matches ? 'Same final state' : 'Different final state'));
    }
    const difference = pair.first_tool_difference;
    if (difference) {
      const names = `${difference.baseline?.name || '—'} → ${difference.candidate?.name || '—'}`;
      const details = node('details');
      details.append(node('summary', `Tool ${difference.index + 1} · ${names}`),
        node('pre', JSON.stringify({active: difference.baseline, candidate: difference.candidate}, null, 2)));
      outcome.append(details);
    }
    if (!outcome.childNodes.length) outcome.textContent = '—';
    row.append(outcome);
    body.append(row);
  }
  table.append(head, body);
  wrap.append(table);
  return wrap;
}

function renderRound(record) {
  const pane = byId('up-round');
  const heading = node('div', undefined, 'up-heading');
  const title = node('div');
  title.append(node('p', `Round ${record.id.slice(0, 12)}`, 'up-provider'),
    node('h2', record.status.replaceAll('_', ' '), 'up-round-status'));
  const artifact = node('a', 'View run data ↗', 'button secondary');
  artifact.href = `${endpoint}/${view.selected}/rounds/${record.id}`;
  heading.append(title, artifact);
  pane.replaceChildren(heading);
  if (record.generation !== view.project.generation && record.status !== 'activated') {
    pane.append(node('p', 'This round belongs to an earlier generation.', 'result-note'));
  }
  if (record.error) pane.append(node('p', record.error, 'error-panel'));
  if (record.proposal) {
    pane.append(node('p', record.proposal.rationale));
    const prompts = node('details');
    prompts.append(node('summary', 'Compare instructions'));
    const columns = node('div', undefined, 'up-prompts');
    for (const [label, text] of [['Active at comparison', record.baseline.system_prompt], ['Candidate', record.proposal.system_prompt]]) {
      const column = node('div');
      column.append(node('h3', label), node('pre', text || '(Empty instructions)'));
      columns.append(column);
    }
    prompts.append(columns);
    pane.append(prompts);
  }
  if (record.decision) {
    pane.append(node('p', record.decision.reason || `${record.decision.repaired_case_ids.length} repaired cases. Candidate passed all required checks.`, 'result-note'));
    pane.append(renderPairs(record.decision.cases));
  }
  if (record.status === 'verified' && record.generation === view.project.generation) {
    const activate = node('button', 'Activate candidate', 'button primary');
    activate.type = 'button';
    activate.addEventListener('click', () => release('activate', activate, record.id));
    pane.append(activate);
  }
}

async function selectRound(id) {
  view.round = id;
  const project = view.selected;
  byId('up-rounds').replaceChildren(...view.project.rounds.map(roundButton));
  byId('up-round').replaceChildren(node('p', 'Loading replay evidence…', 'result-note'));
  try {
    const record = await request(`${endpoint}/${project}/rounds/${id}`);
    if (view.selected === project && view.round === id) renderRound(record);
  } catch (error) {
    if (view.selected === project && view.round === id) notify(error.message, true);
  }
}

async function refresh(project = view.selected) {
  const sequence = ++view.request;
  try {
    const data = await request(endpoint);
    if (sequence !== view.request) return;
    byId('up-connection').textContent = 'Connected';
    const current = data.projects.find(item => item.name === project) || data.projects[0];
    byId('up-empty').hidden = Boolean(current);
    byId('projects').replaceChildren(...data.projects.map(item => {
      const button = node('button', item.name, `job-item${current.name === item.name ? ' selected' : ''}`);
      button.type = 'button';
      button.addEventListener('click', () => refresh(item.name));
      return button;
    }));
    if (!current) {
      byId('up-project').hidden = true;
      return;
    }
    if (view.selected !== current.name) view.round = null;
    view.selected = current.name;
    renderProject(current);
    const selected = current.rounds.find(item => item.id === view.round) || current.rounds[0];
    if (selected) await selectRound(selected.id);
    else byId('up-round').replaceChildren(node('p', 'Replay evidence will appear here after verification.', 'empty-state'));
  } catch (error) {
    if (sequence === view.request) {
      byId('up-connection').textContent = 'Connection unavailable';
      notify(error.message, true);
    }
  }
}

async function release(action, button, round) {
  const project = view.project;
  button.disabled = true;
  const payload = {expected_generation: project.generation};
  if (round) payload.round_id = round;
  try {
    await request(`${endpoint}/${project.name}/${action}`, payload);
    notify(action === 'activate' ? 'Candidate activated for new requests.' : 'Previous recipe restored for new requests.');
    await refresh(project.name);
  } catch (error) {
    notify(error.message, true);
  } finally {
    button.disabled = false;
  }
}

byId('up-refresh').addEventListener('click', () => { notify(''); refresh(); });
byId('up-rollback').addEventListener('click', event => release('rollback', event.currentTarget));
refresh();

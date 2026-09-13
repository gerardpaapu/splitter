import * as FS from 'node:fs/promises';
import {
  stringify,
  stringifyBody,
  stringifyHeadings,
  stringifySegment,
  toInitialChunks,
} from './markdown.ts';
import * as SS from './similarity.ts';
import { wordCount } from './word-count.ts';
import type { Chunk } from './Chunk.ts';

const TARGET_SIZE = 500;
const SIZE_FACTOR = 0.005;
const SIMILARITY_FACTOR = 200;
const BEAMS = 6;
const SPLIT_HEADINGS_FACTOR = 70;
const MAX_ITERATIONS = 20;
const ERROR_THRESHOLD = 1;
const STALE_TOLERANCE = 4;
const DIVERSITY_PENALTY_FACTOR = 800;
const WRITE_WINNER_TO_STDOUT = false;

interface Beam {
  cuts: number[];
  segments: Chunk[][];
  error: number;
  penalty: number;
}

function calculatePenalty(beams: Beam[], idx: number) {
  let total = 0;
  for (let i = 0; i < idx; i++) {
    let len = Math.min(beams[i]!.cuts.length, beams[idx]!.cuts.length);
    for (let j = 0; j < len; j++) {
      if (beams[i]!.cuts[j] === beams[idx]!.cuts[j]) {
        total++;
      }
    }
  }

  return total * DIVERSITY_PENALTY_FACTOR;
}

function dedupeCuts(cutss: number[][]) {
  const result = [] as number[][];
  for (const cuts of cutss) {
    if (result.every((c) => c.join(',') !== cuts.join(','))) {
      result.push(cuts);
    }
  }

  return result;
}

function addRandomBeams(beams: Beam[], chunks: Chunk[]) {
  while (beams.length < BEAMS) {
    let cuts = [];
    let total = 0;

    while (total < chunks.length) {
      let size = Math.floor(Math.random() * (chunks.length - total)) + 1;
      cuts.push(size);
      total += size;
    }

    let segments = applyCuts(cuts, chunks);
    let error = calculateError(segments);

    beams.push({ cuts, segments, error, penalty: 0 });
  }
  beams.sort((a, b) => a.error - b.error);
}

async function main() {
  const txt = await FS.readFile('button.md', 'utf8');
  const chunks = toInitialChunks(txt);

  for (const chunk of chunks) {
    chunk.embedding = await SS.extract(stringifyBody(chunk));
  }

  let iterations = 0;

  let beams = [] as Beam[];
  addRandomBeams(beams, chunks);
  let { segments, cuts, error } = beams[0]!;
  let previous = error;
  let staleCount = 0;

  while (++iterations < MAX_ITERATIONS && error > ERROR_THRESHOLD) {
    beams = dedupeCuts(
      beams.flatMap(({ cuts }) => {
        return mutate(cuts, 2);
      }),
    ).map((mutant) => {
      let candidate = applyCuts(mutant, chunks);
      let error = calculateError(candidate);
      return {
        cuts: mutant,
        segments: candidate,
        error,
        penalty: 0,
      };
    });
    console.log(`evaluated ${beams.length} candidates`);
    beams.sort((a, b) => a.error - b.error);
    beams.forEach((beam, idx) => {
      beam.penalty = calculatePenalty(beams, idx);
    });
    beams.sort((a, b) => a.error + a.penalty - (b.error + b.penalty));
    beams = beams.slice(0, BEAMS);
    error = beams[0]!.error;

    if (error === previous) {
      console.log(`stale: ${++staleCount}`);
    } else {
      staleCount = 0;
      previous = error;
      console.log(
        `error = ${error.toFixed(2).padStart(10)}, iterations = ${iterations}`,
      );
    }

    if (staleCount >= STALE_TOLERANCE) {
      console.log(collectErrors(beams[0]?.segments!));
      console.log(`stale! trying some new randoms`);
      staleCount = 0;
      beams = beams.slice(0, Math.floor(BEAMS / 2));
      addRandomBeams(beams, chunks);
    }
  }
  if (WRITE_WINNER_TO_STDOUT) {
    for (const segment of segments) {
      console.log(stringifySegment(segment));
      console.log('============================');
    }
  }

  console.log(collectErrors(segments));
  console.log(
    `error = ${error.toFixed(2).padStart(10)}, iterations = ${iterations}`,
  );
}

function mutate(cuts: number[], n: number): number[][] {
  let result = [cuts];
  for (let i = 0; i < n; i++) {
    result = result.flatMap(mutateOneStep);
    result = dedupeCuts(result);
  }
  return result;
}

function mutateOneStep(cuts: number[]): number[][] {
  let mutants = [cuts] as number[][];

  // merge one section with the one after it;
  for (let i = 0; i < cuts.length - 1; i++) {
    let mutant = [] as number[];
    for (let j = 0; j < cuts.length; j++) {
      if (i === j) {
        continue;
      }

      if (i + 1 === j) {
        mutant.push(cuts[i]! + cuts[j]!);
      } else {
        mutant.push(cuts[j]!);
      }
    }

    mutants.push(mutant);
  }

  // split one section
  for (let i = 0; i < cuts.length - 1; i++) {
    for (let y = 1; y < cuts[i]!; y++) {
      let mutant = [] as number[];
      for (let j = 0; j < cuts.length; j++) {
        if (j === i) {
          mutant.push(y);
          mutant.push(cuts[i]! - y);
          continue;
        }

        mutant.push(cuts[j]!);
      }

      mutants.push(mutant);
    }
  }

  // grow one section and shrink the one before it
  // shrink one section and grow the one before it
  for (let i = 1; i < cuts.length; i++) {
    const a = [] as number[];
    const b = [] as number[];
    for (let j = 0; j < cuts.length; j++) {
      if (i === j) {
        a.push(cuts[i]! - 1);
        b.push(cuts[i]! + 1);
        continue;
      }

      if (i - 1 === j) {
        a.push(cuts[i - 1]! + 1);
        b.push(cuts[i - 1]! - 1);
        continue;
      }

      a.push(cuts[j]!);
      b.push(cuts[j]!);
    }
    mutants.push(a);
    mutants.push(b);
  }

  return mutants.filter((mutant) => mutant.every((cut) => cut > 0));
}

function applyCuts(cuts: number[], chunks: Chunk[]): Chunk[][] {
  const candidate = [];
  let idx = 0;
  for (const cut of cuts) {
    candidate.push(chunks.slice(idx, idx + cut));
    idx += cut;
  }

  console.assert(idx === chunks.length, `${cuts} = ${chunks.length}`);
  // candidate.push(chunks.slice(idx));
  return candidate;
}
function collectErrors(segments: Chunk[][]) {
  let error = [];

  let sizeDiff = wordCount(stringifySegment(segments[0]!)) - TARGET_SIZE;
  error.push({
    type: 'sizediff',
    raw: sizeDiff,
    error: sizeDiff * sizeDiff * SIZE_FACTOR,
  });

  for (let i = 1; i < segments.length; i++) {
    let previous = segments[i - 1]!;
    let current = segments[i]!;
    let sizeDiff = wordCount(stringifySegment(current)) - TARGET_SIZE;
    error.push({
      type: 'sizediff',
      raw: sizeDiff,
      error: sizeDiff * sizeDiff * SIZE_FACTOR,
    });

    if (current.length === 0 || previous.length === 0) {
      continue;
    }

    let similarity = SS.similarity(
      previous?.at(-1)?.embedding?.data as number[],
      current[0]?.embedding?.data as number[],
    );

    error.push({
      type: 'similarity',
      raw: similarity,
      error: similarity * similarity * SIMILARITY_FACTOR,
    });

    let prevHeadings = stringifyHeadings(previous.at(-1)!);
    let curHeadings = stringifyHeadings(current[0]!);
    let headlinesMatch = curHeadings.startsWith(prevHeadings)
      ? SPLIT_HEADINGS_FACTOR
      : 0;

    error.push({
      type: 'split-headings',
      raw: headlinesMatch,
      error: headlinesMatch,
    });
  }
  return error;
}

function calculateError(segments: Chunk[][]) {
  return collectErrors(segments).reduce((total, { error }) => total + error, 0);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});

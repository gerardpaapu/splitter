import {
  env,
  FeatureExtractionPipeline,
  pipeline,
  cos_sim,
  mean_pooling,
} from '@huggingface/transformers';
import * as process from 'node:process';
import * as PATH from 'node:path';
import * as OS from 'node:os';

let extractorPromise: Promise<FeatureExtractionPipeline> | undefined;

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = process.env.LOCAL_MODEL_PATH ?? PATH.join(OS.homedir(), '.local/share/models');

async function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = pipeline('feature-extraction', 'all-MiniLM-L6-v2', { local_files_only: true });
  }

  return await extractorPromise;
}

// export async function extract(sentence: string) {

//   // TODO: process the whole thing in chunks.
//   const extractor = await getExtractor();
//   const inputs =  extractor.tokenizer(sentence, {
//     truncation: true,
//     max_length: 512,
//   });

//   const output = await extractor.model(inputs);
//   return output.sentence_embedding;
// }

function chunkText(text: string, chunkSize = 1200, overlap = 200): string[] {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    chunks.push(text.slice(start, end));
    
    if (end === text.length) break;
    start += chunkSize - overlap;
  }

  return chunks;
}

export async function extract(documentText: string): Promise<Float32Array> {
  
  const extractor = await getExtractor();
  const chunks = chunkText(documentText); // e.g. 5 chunks

  // 1. Tokenize all chunks into a single batched Tensor [num_chunks, seq_len]
  const inputs = extractor.tokenizer(chunks, {
    padding: true,
    truncation: true,
    max_length: 512,
  });

  // 2. Single ONNX inference call returns sentence_embedding shape [num_chunks, 384]
  const output = await extractor.model(inputs);
  const data = output.sentence_embedding.data as Float32Array;

  const numChunks = chunks.length;
  const dim = 384;
  const combined = new Float32Array(dim);

  for (let c = 0; c < numChunks; c++) {
    const offset = c * dim;
    for (let i = 0; i < dim; i++) {
      combined[i]! += data[offset + i]!;
    }
  }

  let norm = 0;
  for (let i = 0; i < dim; i++) combined[i]! /= numChunks;
  for (let i = 0; i < dim; i++) norm += combined[i]! * combined[i]!;
  norm = Math.sqrt(norm);

  if (norm > 0) {
    for (let i = 0; i < dim; i++) combined[i]! /= norm;
  }
  return combined;
}


export function similarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) {
    dot += a[i]! * b[i]!;
  }
  return dot;
}

// export const similarity = cos_sim;

import {
  FeatureExtractionPipeline,
  pipeline,
  cos_sim,
} from '@huggingface/transformers';
import * as process from 'node:process';

const { MINI_LM_PATH } = process.env;

let extractorPromise: Promise<FeatureExtractionPipeline> | undefined;

export async function extract(sentence: string) {
  extractorPromise = pipeline('feature-extraction', MINI_LM_PATH);
  const extractor = await extractorPromise;
  return await extractor(sentence, { pooling: 'mean', normalize: true });
}

export const similarity = cos_sim;

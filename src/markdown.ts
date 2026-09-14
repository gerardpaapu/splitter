import { remark } from 'remark';
import type { Heading, Root, RootContent, Strong } from 'mdast';
import { removePosition } from 'unist-util-remove-position';
import remarkGfm from 'remark-gfm';
import type { Chunk } from './Chunk.ts';

export type PseudoHeading = Heading | { type: 'paragraph'; children: [Strong] };
function level(heading: PseudoHeading) {
  if (heading.type === 'heading') {
    return heading.depth;
  }
  return 7;
}

export function toInitialChunks(md: string): Chunk[] {
  const chunks = [] as Chunk[];
  const ast = remark().use(remarkGfm).parse(md);
  removePosition(ast);
  const headings = [] as PseudoHeading[];
  let previous: 'heading' | 'text' = 'heading';

  for (const node of ast.children) {
    switch (node.type) {
      case 'heading':
        previous = 'heading';
        while (headings.length && level(headings.at(-1)!) >= level(node)) {
          headings.pop();
        }
        headings.push(node);
        break;

      case 'image':
      case 'table':
      case 'blockquote':
      case 'list':
      case 'code':
        if (chunks.length && previous === 'text') {
          chunks.at(-1)?.figures.push(node);
        } else {
          // if there's a figure after a heading
          // we just have to pretend there was a paragraph
          chunks.push({
            headings: [...headings],
            node: null,
            figures: [node],
          });
          previous = 'text';
        }
        break;

      case 'paragraph':
        if (node.children.length === 1 && node.children[0]?.type === 'strong') {
          while (headings.length && level(headings.at(-1)!) >= 7) {
            headings.pop();
          }

          headings.push(node as any);
          previous = 'heading';

          break;
        }

        chunks.push({
          figures: [],
          headings: [...headings],
          node,
        });
        previous = 'text';

        break;

      default:
        throw new Error(`Not sure what to do with ${node.type}`);
    }
  }

  return chunks;
}

export function stringify(chunk: Chunk) {
  const root = {
    type: 'root',
    children: [
      ...chunk.headings,
      ...(chunk.node ? [chunk.node] : []),
      ...chunk.figures,
    ],
  } as Root;

  return remark().use(remarkGfm).stringify(root);
}

export function stringifyBody(chunk: Chunk) {
  return remark()
    .use(remarkGfm)
    .stringify({
      type: 'root',
      children: [...(chunk.node ? [chunk.node] : []), ...chunk.figures],
    });
}

export function stringifyHeadings(chunk: Chunk) {
  return remark()
    .use(remarkGfm)
    .stringify({
      type: 'root',
      children: [...chunk.headings],
    });
}

export function stringifyHeadingIsh(node: PseudoHeading) {
  return remark()
    .use(remarkGfm)
    .stringify({ type: 'root', children: [node] });
}

export function stringifySegment(segment: Chunk[]): string {
  if (segment.length === 0) {
    return '';
  }

  const first = segment[0]!;
  const root = {
    type: 'root' as const,
    children: [
      ...(first.headings ?? []),
      ...(first.node ? [first.node] : []),
      ...first.figures,
    ],
  } as Root;

  for (let i = 1; i < segment.length; i++) {
    const previous = segment[i - 1]!;
    const current = segment[i]!;
    for (let j = 0; j < current.headings.length; j++) {
      if (previous.headings[j] !== current.headings[j]) {
        root.children.push(...current.headings.slice(j));
        break;
      }
    }
    if (current.node) {
      root.children.push(current.node);
    }
    root.children.push(...current.figures);
  }

  return remark().use(remarkGfm).stringify(root);
}

import type { ResponseFormatTextJSONSchemaConfig } from 'openai/resources/responses/responses';
import { makeParseableTextFormat, type AutoParseableTextFormat } from 'openai/lib/parser';

type JsonSchema =
  | { type: 'string'; description?: string; enum?: readonly string[] }
  | { type: 'number'; description?: string }
  | { type: 'integer'; description?: string }
  | { type: 'array'; description?: string; items: JsonSchema; minItems?: number; maxItems?: number }
  | {
      type: 'object';
      description?: string;
      properties: Record<string, JsonSchema>;
      required?: string[];
      additionalProperties?: boolean;
    };

export interface JotSchema<T> {
  toJsonSchema(): JsonSchema;
  parse(value: unknown, path?: string): T;
}

class StringNode implements JotSchema<string> {
  constructor(readonly options: { description?: string } = {}) {}

  toJsonSchema(): JsonSchema {
    return {
      type: 'string',
      ...(this.options.description ? { description: this.options.description } : {}),
    };
  }

  parse(value: unknown, path: string = 'value'): string {
    if (typeof value !== 'string') {
      throw new TypeError(`${path} must be a string`);
    }

    return value;
  }
}

class EnumNode<TValue extends readonly string[]> implements JotSchema<TValue[number]> {
  constructor(readonly values: TValue, readonly options: { description?: string } = {}) {}

  toJsonSchema(): JsonSchema {
    return {
      type: 'string',
      enum: this.values,
      ...(this.options.description ? { description: this.options.description } : {}),
    };
  }

  parse(value: unknown, path: string = 'value'): TValue[number] {
    if (typeof value !== 'string' || !this.values.includes(value as TValue[number])) {
      throw new TypeError(`${path} must be one of ${this.values.join(', ')}`);
    }

    return value as TValue[number];
  }
}

class ArrayNode<T> implements JotSchema<T[]> {
  constructor(readonly itemNode: JotSchema<T>, readonly options: { description?: string } = {}) {}

  toJsonSchema(): JsonSchema {
    return {
      type: 'array',
      items: this.itemNode.toJsonSchema(),
      ...(this.options.description ? { description: this.options.description } : {}),
    };
  }

  parse(value: unknown, path: string = 'value'): T[] {
    if (!Array.isArray(value)) {
      throw new TypeError(`${path} must be an array`);
    }

    return value.map((item, index) => this.itemNode.parse(item, `${path}[${index}]`));
  }
}

export interface ObjectNodeOptions {
  description?: string;
  allowAdditionalProperties?: boolean;
}

class ObjectNode<Shape extends Record<string, JotSchema<any>>> implements JotSchema<{ [K in keyof Shape]: InferJot<Shape[K]> }> {
  constructor(readonly shape: Shape, readonly options: ObjectNodeOptions = {}) {}

  toJsonSchema(): JsonSchema {
    const properties: Record<string, JsonSchema> = {};
    const required: string[] = [];

    for (const [key, node] of Object.entries(this.shape)) {
      properties[key] = node.toJsonSchema();
      required.push(key);
    }

    return {
      type: 'object',
      properties,
      required,
      additionalProperties: this.options.allowAdditionalProperties ?? false,
      ...(this.options.description ? { description: this.options.description } : {}),
    };
  }

  parse(value: unknown, path: string = 'value') {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      throw new TypeError(`${path} must be an object`);
    }

    const result: Record<string, unknown> = {};
    for (const key of Object.keys(this.shape)) {
      const node = this.shape[key]!;
      const nextPath = `${path}.${key}`;
      result[key] = node.parse((value as Record<string, unknown>)[key], nextPath);
    }

    return result as { [K in keyof Shape]: InferJot<Shape[K]> };
  }
}

export type InferJot<TSchema> = TSchema extends JotSchema<infer TValue> ? TValue : never;

export const jot = {
  string: (options?: { description?: string }): JotSchema<string> => new StringNode(options),
  enum: <TValue extends readonly string[]>(values: TValue, options?: { description?: string }): JotSchema<TValue[number]> =>
    new EnumNode(values, options),
  array: <T>(schema: JotSchema<T>, options?: { description?: string }): JotSchema<T[]> =>
    new ArrayNode(schema, options),
  object: <Shape extends Record<string, JotSchema<any>>>(shape: Shape, options?: ObjectNodeOptions) =>
    new ObjectNode(shape, options),
};

export function compileJotSchema<T>(name: string, schema: JotSchema<T>): AutoParseableTextFormat<T> {
  const format: ResponseFormatTextJSONSchemaConfig = {
    type: 'json_schema',
    name,
    schema: schema.toJsonSchema(),
  };

  return makeParseableTextFormat(format, (raw) => schema.parse(JSON.parse(raw)));
}

export interface InputWrapper {
  id: number;
  name: string;
  experiment_id: number;
  index: number;
  config: { [key: string]: number };
}

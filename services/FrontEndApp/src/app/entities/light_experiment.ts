import { System } from '../entities/system'
import { Explorer } from '../entities/explorer'
import { InputWrapper } from '../entities/input_wrapper'
import { OutputRepresentation } from '../entities/output_representation'

export interface LightExperiment {
    id: number;
    name: string;
    created_on: Date;
    progress: number;
    systems: System[];
    explorers: Explorer[];
    input_wrappers: InputWrapper[];
    output_representations: OutputRepresentation[];
    status: number;
  }
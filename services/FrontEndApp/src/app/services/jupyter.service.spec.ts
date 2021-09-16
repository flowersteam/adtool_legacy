import { TestBed } from '@angular/core/testing';

import { JupyterService } from './jupyter.service';

describe('JupyterService', () => {
  let service: JupyterService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(JupyterService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

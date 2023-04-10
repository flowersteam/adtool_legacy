import { TestBed } from '@angular/core/testing';

import { FiletypeConverterService } from './filetype-converter.service';

describe('FiletypeConverterService', () => {
  let service: FiletypeConverterService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(FiletypeConverterService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

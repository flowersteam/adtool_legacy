import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LoadExperimentConfigToCreateComponent } from './load-experiment-config-to-create.component';

describe('LoadExperimentConfigToCreateComponent', () => {
  let component: LoadExperimentConfigToCreateComponent;
  let fixture: ComponentFixture<LoadExperimentConfigToCreateComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LoadExperimentConfigToCreateComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(LoadExperimentConfigToCreateComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

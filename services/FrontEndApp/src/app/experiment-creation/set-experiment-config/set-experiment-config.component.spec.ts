import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SetExperimentConfigComponent } from './set-experiment-config.component';

describe('SetExperimentConfigComponent', () => {
  let component: SetExperimentConfigComponent;
  let fixture: ComponentFixture<SetExperimentConfigComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SetExperimentConfigComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(SetExperimentConfigComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

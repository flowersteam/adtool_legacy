import { ComponentFixture, TestBed } from '@angular/core/testing';

import { JupyterLabComponent } from './jupyter-lab.component';

describe('JupyterLabComponent', () => {
  let component: JupyterLabComponent;
  let fixture: ComponentFixture<JupyterLabComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [JupyterLabComponent],
    }).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(JupyterLabComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

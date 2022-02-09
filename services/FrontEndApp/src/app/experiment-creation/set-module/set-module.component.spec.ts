import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SetModuleComponent } from './set-module.component';

describe('SetModuleComponent', () => {
  let component: SetModuleComponent;
  let fixture: ComponentFixture<SetModuleComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SetModuleComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(SetModuleComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SetDictionaryComponent } from './set-dictionary.component';

describe('SetDictionaryComponent', () => {
  let component: SetDictionaryComponent;
  let fixture: ComponentFixture<SetDictionaryComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SetDictionaryComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(SetDictionaryComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

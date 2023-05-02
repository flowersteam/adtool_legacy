import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-code-editor',
  templateUrl: './code-editor.component.html',
  styleUrls: ['./code-editor.component.scss'],
})
export class CodeEditorComponent implements OnInit {
  @Input() inputValue?: any;
  @Output() inputValueChange = new EventEmitter();
  @Output() triggerParentMethod = new EventEmitter<any>();

  constructor() {}

  editorOptions = {
    theme: 'vs-dark',
    language: 'python',
    automaticLayout: true,
  };
  code: string = '';

  ngOnInit(): void {
    this.code = JSON.stringify(this.inputValue);
  }

  returnToParent() {
    this.inputValueChange.emit(this.code);
    this.triggerParentMethod.next();
  }
}
